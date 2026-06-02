// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build unix

package wasip1

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
)

// maxSymlinkDepth is the maximum number of symlink resolutions allowed.
const maxSymlinkDepth = 40

// defaultFileMode is the default permission mode for newly created files.
const defaultFileMode = 0o600

// utimeNow and utimeOmit are special values for Timespec.Nsec.
var (
	utimeNow  int64 // Set to current time
	utimeOmit int64 // Don't change
)

func init() {
	switch runtime.GOOS {
	case "linux":
		// https://github.com/torvalds/linux/blob/master/include/linux/stat.h#L15-L16
		utimeNow = (1 << 30) - 1
		utimeOmit = (1 << 30) - 2
	case "darwin":
		// https://github.com/apple/darwin-xnu/blob/main/bsd/sys/stat.h#L575-L576
		utimeNow = -1
		utimeOmit = -2
	case "openbsd":
		// https://github.com/openbsd/src/blob/master/sys/sys/stat.h#L188-L189
		utimeNow = -2
		utimeOmit = -1
	default:
		// Most (all?) other UNIXes use -1/-2, e.g. FreeBSD:
		// https://github.com/freebsd/freebsd-src/blob/main/sys/sys/stat.h#L359-L360
		utimeNow = -1
		utimeOmit = -2
	}
}

// hostFile is the default File implementation, backed by an *os.File.
type hostFile struct {
	*os.File
	// appendMode tracks whether the file was opened (or reconfigured) with
	// O_APPEND, so WriteAt can still honor an explicit offset.
	appendMode bool
}

// NewHostFile wraps a host *os.File as a File. It is the bridge for using
// process streams (e.g. os.Stdin) or other host files with the builder's
// backend-agnostic WithStdin/WithStdout/WithStderr methods.
func NewHostFile(f *os.File) (File, error) {
	return &hostFile{File: f}, nil
}

// WriteAt writes at the given offset. os.File.WriteAt rejects files opened with
// O_APPEND, so for those we issue a pwrite, which writes at the offset
// regardless of the append flag.
func (f *hostFile) WriteAt(p []byte, off int64) (int, error) {
	if f.appendMode {
		return unix.Pwrite(int(f.Fd()), p, off)
	}
	return f.File.WriteAt(p, off)
}

func (f *hostFile) FileStat() (FileStat, error) {
	var st unix.Stat_t
	if err := unix.Fstat(int(f.Fd()), &st); err != nil {
		return FileStat{}, err
	}
	return statFromUnix(&st), nil
}

func (f *hostFile) SetFlags(appendFlag, nonblock bool) error {
	var osFlags int
	if appendFlag {
		osFlags |= unix.O_APPEND
	}
	if nonblock {
		osFlags |= unix.O_NONBLOCK
	}
	if _, err := unix.FcntlInt(f.Fd(), unix.F_SETFL, osFlags); err != nil {
		return err
	}
	f.appendMode = appendFlag
	return nil
}

func (f *hostFile) SetTimes(atim, mtim int64, fstFlags int32) error {
	times, err := buildTimespec(atim, mtim, fstFlags)
	if err != nil {
		return err
	}
	// Use /dev/fd/N to reference the open file descriptor, avoiding TOCTOU races
	// while maintaining nanosecond precision.
	path := fmt.Sprintf("/dev/fd/%d", f.Fd())
	return unix.UtimesNanoAt(unix.AT_FDCWD, path, times, 0)
}

func (f *hostFile) Accept() (File, error) {
	nfd, _, err := unix.Accept(int(f.Fd()))
	if err != nil {
		return nil, err
	}
	return &hostFile{File: os.NewFile(uintptr(nfd), "")}, nil
}

func (f *hostFile) Shutdown(how int32) error {
	switch how {
	case shutRd:
		return unix.Shutdown(int(f.Fd()), unix.SHUT_RD)
	case shutWr:
		return unix.Shutdown(int(f.Fd()), unix.SHUT_WR)
	case shutRdWr:
		return unix.Shutdown(int(f.Fd()), unix.SHUT_RDWR)
	default:
		return syscall.EINVAL
	}
}

// hostFileSystem is the default FileSystem implementation. It is rooted at an
// open directory handle and resolves every path relative to that directory
// using *at syscalls, never escaping it: ".." is traversed at runtime and
// symlinks are resolved within the sandbox (see resolvePath).
type hostFileSystem struct {
	handle *hostFile // the directory's own open handle (the sandbox root)
}

// OpenHostFileSystem opens dir as a sandboxed FileSystem. Subsequent
// operations are confined to that directory tree.
func OpenHostFileSystem(dir string) (FileSystem, error) {
	f, err := os.OpenFile(dir, os.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	return &hostFileSystem{handle: &hostFile{File: f}}, nil
}

// dir returns the directory handle every path is resolved relative to.
func (h *hostFileSystem) dir() *os.File { return h.handle.File }

func (h *hostFileSystem) Open(name string) (fs.File, error) {
	return openat(h.dir(), name, true, 0, 0, uint64(RightsFdRead))
}

func (h *hostFileSystem) Handle() File { return h.handle }

func (h *hostFileSystem) Close() error { return h.handle.Close() }

func (h *hostFileSystem) OpenFile(
	name string,
	followSymlink bool,
	oflags, fdflags int32,
	rights uint64,
) (File, error) {
	f, err := openat(h.dir(), name, followSymlink, oflags, fdflags, rights)
	if err != nil {
		return nil, err
	}
	appendMode := fdflags&int32(fdFlagsAppend) != 0
	return &hostFile{File: f, appendMode: appendMode}, nil
}

func (h *hostFileSystem) OpenRoot(name string) (FileSystem, error) {
	f, err := openat(
		h.dir(), name, true, int32(oFlagsDirectory), 0, uint64(RightsFdRead),
	)
	if err != nil {
		return nil, err
	}
	return &hostFileSystem{handle: &hostFile{File: f}}, nil
}

func (h *hostFileSystem) Mkdir(name string, perm fs.FileMode) error {
	return mkdirat(h.dir(), name, uint32(perm))
}

func (h *hostFileSystem) Stat(name string, followSymlink bool) (FileStat, error) {
	return stat(h.dir(), name, followSymlink)
}

func (h *hostFileSystem) Readlink(name string) (string, error) {
	return readlink(h.dir(), name)
}

func (h *hostFileSystem) Symlink(target, name string) error {
	return symlinkat(target, h.dir(), name)
}

func (h *hostFileSystem) Unlink(name string) error {
	return unlinkat(h.dir(), name)
}

func (h *hostFileSystem) Rmdir(name string) error {
	return rmdirat(h.dir(), name)
}

func (h *hostFileSystem) Rename(
	newDir FileSystem,
	oldName, newName string,
) error {
	target, ok := newDir.(*hostFileSystem)
	if !ok {
		return syscall.EXDEV
	}
	return renameat(h.dir(), oldName, target.dir(), newName)
}

func (h *hostFileSystem) Link(
	oldName string,
	followSymlink bool,
	newDir FileSystem,
	newName string,
) error {
	target, ok := newDir.(*hostFileSystem)
	if !ok {
		return syscall.EXDEV
	}
	return linkat(h.dir(), oldName, followSymlink, target.dir(), newName)
}

func (h *hostFileSystem) Chtimes(
	name string,
	atim, mtim int64,
	fstFlags int32,
	followSymlink bool,
) error {
	return utimes(h.dir(), name, atim, mtim, fstFlags, followSymlink)
}

// ReadDir reads directory entries from the root, returning synthetic "." and
// ".." entries followed by actual directory content, as required by
// fd_readdir.
//
// For each entry, the inode is obtained via Fstatat. The "." entry uses the
// directory's own inode; ".." uses inode 0 because we cannot safely access the
// parent directory due to sandboxing.
func (h *hostFileSystem) ReadDir() ([]DirEntry, error) {
	dir := h.dir()

	var dirStat unix.Stat_t
	if err := unix.Fstat(int(dir.Fd()), &dirStat); err != nil {
		return nil, err
	}

	// Seek to start to ensure we read all entries.
	if _, err := dir.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	entries, err := dir.ReadDir(-1)
	if err != nil {
		return nil, err
	}

	result := make([]DirEntry, 0, len(entries)+2)
	result = append(
		result,
		DirEntry{Name: ".", FileType: FileTypeDirectory, Ino: dirStat.Ino},
		DirEntry{Name: "..", FileType: FileTypeDirectory, Ino: 0},
	)

	dirFd := int(dir.Fd())
	for _, entry := range entries {
		var statBuf unix.Stat_t
		err := unix.Fstatat(dirFd, entry.Name(), &statBuf, unix.AT_SYMLINK_NOFOLLOW)
		if err != nil {
			return nil, err
		}
		result = append(result, DirEntry{
			Name:     entry.Name(),
			FileType: fileTypeFromMode(uint32(statBuf.Mode)),
			Ino:      statBuf.Ino,
		})
	}
	return result, nil
}

// mkdirat creates a directory relative to dir.
func mkdirat(dir *os.File, path string, mode uint32) error {
	if !isRelativePath(path) {
		return os.ErrInvalid
	}

	components, err := getComponents(path)
	if err != nil {
		return err
	}

	if len(components) == 1 && components[0] == "." {
		return os.ErrInvalid
	}

	parentFd, _, _, err := walkToParent(dir, components, 0)
	if err != nil {
		return err
	}

	if parentFd != int(dir.Fd()) {
		defer unix.Close(parentFd)
	}

	return unix.Mkdirat(parentFd, components[len(components)-1], mode)
}

// stat returns the FileStat of a file or directory relative to dir.
// followSymlinks controls whether the final symlink is followed.
func stat(dir *os.File, path string, followSymlinks bool) (FileStat, error) {
	dirFd, fileName, err := resolvePath(dir, path, followSymlinks, 0)
	if err != nil {
		return FileStat{}, err
	}
	if dirFd != int(dir.Fd()) {
		defer unix.Close(dirFd)
	}

	var statBuf unix.Stat_t
	if err := unix.Fstatat(
		dirFd, fileName, &statBuf, unix.AT_SYMLINK_NOFOLLOW,
	); err != nil {
		return FileStat{}, err
	}
	return statFromUnix(&statBuf), nil
}

// utimes sets the access and modification times of a file or directory.
func utimes(
	dir *os.File,
	path string,
	atim, mtim int64,
	fstFlags int32,
	followSymlinks bool,
) error {
	times, err := buildTimespec(atim, mtim, fstFlags)
	if err != nil {
		return err
	}

	dirFd, fileName, err := resolvePath(dir, path, followSymlinks, 0)
	if err != nil {
		return err
	}
	if dirFd != int(dir.Fd()) {
		defer unix.Close(dirFd)
	}

	return unix.UtimesNanoAt(dirFd, fileName, times, unix.AT_SYMLINK_NOFOLLOW)
}

// linkat creates a hard link to an existing file.
func linkat(
	oldDir *os.File,
	oldPath string,
	followSymlinks bool,
	newDir *os.File,
	newPath string,
) error {
	// Hard link creation does not support directory targets (implied by trailing
	// slash).
	if strings.HasSuffix(oldPath, string(filepath.Separator)) ||
		strings.HasSuffix(newPath, string(filepath.Separator)) {
		return syscall.ENOENT
	}

	oldDirFd, oldName, err := resolvePath(oldDir, oldPath, followSymlinks, 0)
	if err != nil {
		return err
	}
	if oldDirFd != int(oldDir.Fd()) {
		defer unix.Close(oldDirFd)
	}

	newDirFd, newName, err := resolvePath(newDir, newPath, false, 0)
	if err != nil {
		return err
	}
	if newDirFd != int(newDir.Fd()) {
		defer unix.Close(newDirFd)
	}

	return unix.Linkat(oldDirFd, oldName, newDirFd, newName, 0)
}

// readlink reads the contents of a symbolic link relative to dir.
func readlink(dir *os.File, path string) (string, error) {
	dirFd, name, err := resolvePath(dir, path, false, 0)
	if err != nil {
		return "", err
	}
	if dirFd != int(dir.Fd()) {
		defer unix.Close(dirFd)
	}
	return readlinkat(dirFd, name)
}

// rmdirat removes an empty directory relative to dir.
func rmdirat(dir *os.File, path string) error {
	dirFd, name, err := resolvePath(dir, path, false, 0)
	if err != nil {
		return err
	}
	if dirFd != int(dir.Fd()) {
		defer unix.Close(dirFd)
	}
	return unix.Unlinkat(dirFd, name, unix.AT_REMOVEDIR)
}

// renameat renames a file or directory.
func renameat(
	oldDir *os.File,
	oldPath string,
	newDir *os.File,
	newPath string,
) error {
	oldDirFd, oldName, err := resolvePath(oldDir, oldPath, false, 0)
	if err != nil {
		return err
	}
	if oldDirFd != int(oldDir.Fd()) {
		defer unix.Close(oldDirFd)
	}

	newDirFd, newName, err := resolvePath(newDir, newPath, false, 0)
	if err != nil {
		return err
	}
	if newDirFd != int(newDir.Fd()) {
		defer unix.Close(newDirFd)
	}

	return unix.Renameat(oldDirFd, oldName, newDirFd, newName)
}

// symlinkat creates a symbolic link at path pointing to target.
func symlinkat(target string, dir *os.File, path string) error {
	if strings.HasPrefix(target, string(filepath.Separator)) {
		return syscall.EPERM
	}
	if strings.HasSuffix(path, string(filepath.Separator)) {
		return syscall.ENOENT
	}

	dirFd, name, err := resolvePath(dir, path, false, 0)
	if err != nil {
		return err
	}
	if dirFd != int(dir.Fd()) {
		defer unix.Close(dirFd)
	}

	return unix.Symlinkat(target, dirFd, name)
}

// unlinkat removes a file (but not a directory). It returns EISDIR if the path
// refers to a directory.
func unlinkat(dir *os.File, path string) error {
	dirFd, name, err := resolvePath(dir, path, false, 0)
	if err != nil {
		return err
	}
	if dirFd != int(dir.Fd()) {
		defer unix.Close(dirFd)
	}

	// Restore a trailing slash so the syscall returns the correct error.
	if strings.HasSuffix(path, string(filepath.Separator)) {
		name += string(filepath.Separator)
	}

	return unix.Unlinkat(dirFd, name, 0)
}

// openat opens a file or directory relative to dir, securely resolving the
// path within the sandbox. followSymlinks controls whether the final symlink
// is followed.
func openat(
	dir *os.File,
	path string,
	followSymlinks bool,
	oflags, fdflags int32,
	fsRights uint64,
) (*os.File, error) {
	// A trailing slash means the path must be a directory.
	// We detect this before calling resolvePath because we lose that info.
	if strings.HasSuffix(path, string(filepath.Separator)) {
		oflags |= int32(oFlagsDirectory)
	}

	dirFd, name, err := resolvePath(dir, path, followSymlinks, 0)
	if err != nil {
		return nil, err
	}

	var parentDir *os.File
	if dirFd == int(dir.Fd()) {
		parentDir = dir
	} else {
		parentDir = os.NewFile(uintptr(dirFd), "")
		defer parentDir.Close()
	}

	// Determine read/write mode from rights.
	canRead := fsRights&uint64(RightsFdRead) != 0
	canWrite := fsRights&uint64(RightsFdWrite) != 0

	flags := unix.O_CLOEXEC | unix.O_NOFOLLOW
	switch {
	case canRead && canWrite:
		flags |= unix.O_RDWR
	case canWrite:
		flags |= unix.O_WRONLY
	default:
		flags |= unix.O_RDONLY
	}

	// Convert WASI oflags to Unix flags.
	if oflags&int32(oFlagsCreat) != 0 {
		flags |= unix.O_CREAT
	}
	if oflags&int32(oFlagsDirectory) != 0 {
		flags |= unix.O_DIRECTORY
	}
	if oflags&int32(oFlagsExcl) != 0 {
		flags |= unix.O_EXCL
	}
	if oflags&int32(oFlagsTrunc) != 0 {
		flags |= unix.O_TRUNC
	}

	// Convert WASI fdflags to Unix flags.
	if fdflags&int32(fdFlagsAppend) != 0 {
		flags |= unix.O_APPEND
	}
	if fdflags&int32(fdFlagsDsync) != 0 {
		flags |= unix.O_DSYNC
	}
	if fdflags&int32(fdFlagsNonblock) != 0 {
		flags |= unix.O_NONBLOCK
	}
	if fdflags&int32(fdFlagsSync) != 0 {
		flags |= unix.O_SYNC
	}
	// fdFlagsRsync maps to O_RSYNC, which equals O_SYNC on most systems.
	if fdflags&int32(fdFlagsRsync) != 0 {
		flags |= unix.O_SYNC
	}

	fd, err := unix.Openat(int(parentDir.Fd()), name, flags, defaultFileMode)
	if err != nil {
		return nil, err
	}

	return os.NewFile(uintptr(fd), name), nil
}

// walkToParent walks through intermediate path components (all except the last)
// and returns the fd of the parent directory of the final component.
// Symlinks in intermediate components are followed securely within the sandbox.
//
// The caller is responsible for closing parentFd if it differs from dir.Fd().
func walkToParent(
	dir *os.File,
	components []string,
	depth int,
) (int, string, int, error) {
	if depth >= maxSymlinkDepth {
		return 0, "", depth, syscall.ELOOP
	}

	dirFd := int(dir.Fd())
	parentFd := dirFd
	parentPath := ""

	// Helper to close parentFd if it differs from dir.Fd.
	closeParent := func() {
		if parentFd != dirFd {
			unix.Close(parentFd)
		}
	}

	// Re-calculates path components and restarts the walk from root.
	restart := func(newBase string, remain []string) (int, string, int, error) {
		closeParent()

		newComponents := append(splitPath(newBase), remain...)
		if len(newComponents) == 0 {
			newComponents = []string{"."}
		}

		return walkToParent(dir, newComponents, depth+1)
	}

	for i, component := range components[:len(components)-1] {
		if component == "." {
			continue
		}

		if component == ".." {
			// Check if we're at the root, we cannot go above sandbox.
			if parentPath == "" {
				return 0, "", depth, syscall.EPERM
			}
			newParentPath := filepath.Dir(parentPath)
			if newParentPath == "." {
				newParentPath = ""
			}
			return restart(newParentPath, components[i+1:])
		}

		mode := unix.O_RDONLY | unix.O_NOFOLLOW | unix.O_DIRECTORY | unix.O_CLOEXEC
		newFd, err := unix.Openat(parentFd, component, mode, 0)

		// If Openat fails, check if the component is a symlink we need to resolve.
		// O_NOFOLLOW typically returns ELOOP for symlinks. However, when combined
		// with O_DIRECTORY, some systems may return ENOTDIR if the symlink points
		// to a non-directory file.
		if errors.Is(err, syscall.ELOOP) || errors.Is(err, syscall.ENOTDIR) {
			resolvedPath, symErr := resolveSymlink(parentFd, parentPath, component)
			if symErr != nil {
				closeParent()
				// If it's not a symlink, return the original error.
				return 0, "", depth, err
			}
			return restart(resolvedPath, components[i+1:])
		}

		closeParent()
		if err != nil {
			return 0, "", depth, err
		}

		parentFd = newFd
		parentPath = filepath.Join(parentPath, component)
	}

	return parentFd, parentPath, depth, nil
}

func getComponents(path string) ([]string, error) {
	components := splitPath(path)
	if len(components) == 0 {
		return nil, os.ErrInvalid
	}

	// If the final component is "..", append "." to ensure it's processed as an
	// intermediate component through safe logical resolution, and we use "." as
	// the final name for the syscall.
	if components[len(components)-1] == ".." {
		components = append(components, ".")
	}

	return components, nil
}

// splitPath splits a path into its components without lexically resolving "..".
// This is critical for security: ".." must be traversed at runtime to enforce
// permission and existence checks on intermediate directories.
// It handles both forward slashes and the OS-specific separator.
func splitPath(path string) []string {
	parts := strings.FieldsFunc(path, func(r rune) bool {
		return r == filepath.Separator
	})
	if len(parts) == 0 {
		return []string{"."}
	}
	return parts
}

// isRelativePath checks if a path is a valid relative path for sandbox
// traversal. It rejects:
//   - Absolute paths (starting with /)
//   - Paths that start with .. (immediate sandbox escape)
//   - Empty paths
//
// Unlike filepath.IsLocal, it ALLOWS internal ".." components like "a/../b"
// because these will be traversed at runtime, enforcing permission checks.
func isRelativePath(path string) bool {
	if path == "" {
		return false
	}
	if filepath.IsAbs(path) {
		return false
	}
	// Check if path starts with "..".
	if path == ".." || strings.HasPrefix(path, ".."+string(filepath.Separator)) {
		return false
	}
	return true
}

// resolveSymlink reads a symlink and returns a sandbox-safe resolved path.
//
// Absolute symlink targets are resolved relative to the sandbox root (not the
// filesystem root), while relative targets are resolved relative to parentPath.
// Returns EPERM if the resolved path would escape the sandbox.
func resolveSymlink(parentFd int, parentPath, name string) (string, error) {
	target, err := readlinkat(parentFd, name)
	if err != nil {
		return "", err
	}

	var resolved string
	if filepath.IsAbs(target) {
		resolved = strings.TrimPrefix(target, string(filepath.Separator))
	} else {
		resolved = filepath.Clean(filepath.Join(parentPath, target))
	}
	if !filepath.IsLocal(resolved) {
		return "", syscall.EPERM
	}
	return resolved, nil
}

// readlinkat reads a symlink target, growing the buffer as needed.
func readlinkat(dirFd int, name string) (string, error) {
	buf := make([]byte, 256)
	for {
		n, err := unix.Readlinkat(dirFd, name, buf)
		if err != nil {
			return "", err
		}
		if n < len(buf) {
			return string(buf[:n]), nil
		}
		buf = make([]byte, len(buf)*2)
	}
}

// resolvePath resolves a path relative to a directory os.File. It returns the
// file descriptor of the parent directory and the file or directory name.
//
// It does so by securely resolving path relative to dirFd, following symlinks
// in intermediate components and, if followSymlinks is true, in the final
// component.
//
// If the final path component does not exist, it returns success (and ENOENT
// checks are deferred to the caller), which allows support for O_CREAT.
//
// All of this is done while never escaping the sandbox root.
func resolvePath(
	dir *os.File,
	path string,
	followSymlinks bool,
	depth int,
) (int, string, error) {
	if depth >= maxSymlinkDepth {
		return 0, "", syscall.ELOOP
	}

	if !isRelativePath(path) {
		return 0, "", syscall.EPERM
	}

	comps, err := getComponents(path)
	if err != nil {
		return 0, "", err
	}

	// Handle special case of stat on "." (the directory itself).
	if len(comps) == 1 && comps[0] == "." {
		return int(dir.Fd()), ".", nil
	}

	parentFd, parentPath, newDepth, err := walkToParent(dir, comps, depth)
	if err != nil {
		return 0, "", err
	}

	// closeParent closes parentFd if we own it (differs from dir.Fd). We cannot
	// use defer because we may return parentFd to the caller.
	closeParent := func() {
		if parentFd != int(dir.Fd()) {
			unix.Close(parentFd)
		}
	}

	finalName := comps[len(comps)-1]

	// Always stat with AT_SYMLINK_NOFOLLOW first to check if it's a symlink.
	var statBuf unix.Stat_t
	err = unix.Fstatat(parentFd, finalName, &statBuf, unix.AT_SYMLINK_NOFOLLOW)

	// If the error is ENOENT, the file doesn't exist. We return success (for
	// O_CREAT support).
	if errors.Is(err, syscall.ENOENT) {
		return parentFd, finalName, nil
	}

	if err != nil {
		closeParent()
		return 0, "", err
	}

	// If it's not a symlink, or we don't want to follow, return immediately.
	if statBuf.Mode&unix.S_IFMT != unix.S_IFLNK || !followSymlinks {
		return parentFd, finalName, nil
	}

	// It's a symlink and we need to follow it securely.
	resolvedPath, err := resolveSymlink(parentFd, parentPath, finalName)
	closeParent()
	if err != nil {
		return 0, "", err
	}

	// Restart resolution with the new target and updated depth.
	return resolvePath(dir, resolvedPath, followSymlinks, newDepth+1)
}

// statFromUnix converts a unix.Stat_t to a FileStat.
func statFromUnix(s *unix.Stat_t) FileStat {
	return FileStat{
		Dev:      uint64(s.Dev),
		Ino:      s.Ino,
		FileType: fileTypeFromMode(uint32(s.Mode)),
		Nlink:    uint64(s.Nlink),
		Size:     uint64(s.Size),
		Atim:     uint64(unix.TimespecToNsec(s.Atim)),
		Mtim:     uint64(unix.TimespecToNsec(s.Mtim)),
		Ctim:     uint64(unix.TimespecToNsec(s.Ctim)),
	}
}

// fileTypeFromMode extracts the WASI file type from a Unix mode.
func fileTypeFromMode(mode uint32) FileType {
	switch mode & unix.S_IFMT {
	case unix.S_IFBLK:
		return FileTypeBlockDevice
	case unix.S_IFCHR:
		return FileTypeCharacterDevice
	case unix.S_IFDIR:
		return FileTypeDirectory
	case unix.S_IFREG:
		return FileTypeRegularFile
	case unix.S_IFSOCK:
		return FileTypeSocketStream
	case unix.S_IFLNK:
		return FileTypeSymbolicLink
	default:
		return FileTypeUnknown
	}
}

func buildTimespec(atim, mtim int64, fstFlags int32) ([]unix.Timespec, error) {
	// ATIM and ATIM_NOW are mutually exclusive, as are MTIM and MTIM_NOW.
	if (fstFlags&fstFlagsAtim != 0 && fstFlags&fstFlagsAtimNow != 0) ||
		(fstFlags&fstFlagsMtim != 0 && fstFlags&fstFlagsMtimNow != 0) {
		return nil, syscall.EINVAL
	}

	var atimSpec unix.Timespec
	switch {
	case fstFlags&fstFlagsAtimNow != 0:
		atimSpec = unix.Timespec{Nsec: utimeNow}
	case fstFlags&fstFlagsAtim != 0:
		atimSpec = unix.NsecToTimespec(atim)
	default:
		atimSpec = unix.Timespec{Nsec: utimeOmit}
	}

	var mtimSpec unix.Timespec
	switch {
	case fstFlags&fstFlagsMtimNow != 0:
		mtimSpec = unix.Timespec{Nsec: utimeNow}
	case fstFlags&fstFlagsMtim != 0:
		mtimSpec = unix.NsecToTimespec(mtim)
	default:
		mtimSpec = unix.Timespec{Nsec: utimeOmit}
	}

	return []unix.Timespec{atimSpec, mtimSpec}, nil
}
