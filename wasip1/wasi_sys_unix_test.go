// Copyright 2025 Google LLC
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
	"io"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// fsEntry describes a filesystem entry to create in a test.
type fsEntry struct {
	path    string // relative path from root
	content string // file content (ignored for dirs/links)
	link    string // symlink target (if non-empty, creates a symlink)
	isDir   bool   // true = directory
}

func dir(path string) fsEntry {
	return fsEntry{path: path, isDir: true}
}

func file(path, content string) fsEntry {
	return fsEntry{path: path, content: content}
}

func link(path, target string) fsEntry {
	return fsEntry{path: path, link: target}
}

// testFS creates a filesystem structure for testing and returns the root
// directory path. Parent directories are created automatically.
func testFS(t *testing.T, entries ...fsEntry) string {
	t.Helper()
	root := t.TempDir()

	for _, e := range entries {
		fullPath := filepath.Join(root, e.path)
		parentDir := filepath.Dir(fullPath)

		// Ensure parent directories exist
		if err := os.MkdirAll(parentDir, 0o755); err != nil {
			t.Fatalf("failed to create parent dirs for %s: %v", e.path, err)
		}

		switch {
		case e.link != "":
			if err := os.Symlink(e.link, fullPath); err != nil {
				t.Fatalf("failed to create symlink %s: %v", e.path, err)
			}
		case e.isDir:
			if err := os.Mkdir(fullPath, 0o755); err != nil {
				t.Fatalf("failed to create dir %s: %v", e.path, err)
			}
		default:
			if err := os.WriteFile(fullPath, []byte(e.content), 0o644); err != nil {
				t.Fatalf("failed to create file %s: %v", e.path, err)
			}
		}
	}

	return root
}

// testHostFS builds the default host FileSystem over a temporary directory
// populated with entries. The returned FileSystem is closed automatically when
// the test finishes.
func testHostFS(t *testing.T, entries ...fsEntry) (string, FileSystem) {
	t.Helper()
	root := testFS(t, entries...)
	fsys, err := OpenHostFileSystem(root)
	if err != nil {
		t.Fatalf("OpenHostFileSystem failed: %v", err)
	}
	t.Cleanup(func() { fsys.Close() })
	return root, fsys
}

func TestOpenFile_BasicFile(t *testing.T) {
	_, fsys := testHostFS(t, file("test.txt", "hello"))

	f, err := fsys.OpenFile("test.txt", false, 0, 0, uint64(RightsFdRead))
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}
	if string(content) != "hello" {
		t.Errorf("got %q, want %q", string(content), "hello")
	}
}

func TestOpenFile_NestedPath(t *testing.T) {
	_, fsys := testHostFS(t, file("a/b/c/test.txt", "nested"))

	f, err := fsys.OpenFile("a/b/c/test.txt", false, 0, 0, uint64(RightsFdRead))
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}
	if string(content) != "nested" {
		t.Errorf("got %q, want %q", string(content), "nested")
	}
}

func TestOpenRoot_Directory(t *testing.T) {
	_, fsys := testHostFS(t, dir("subdir"))

	sub, err := fsys.OpenRoot("subdir")
	if err != nil {
		t.Fatalf("OpenRoot failed: %v", err)
	}
	defer sub.Close()

	info, err := sub.Handle().Stat()
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if !info.IsDir() {
		t.Error("expected directory")
	}
}

func TestOpenRoot_OnFileFails(t *testing.T) {
	_, fsys := testHostFS(t, file("file.txt", "content"))

	sub, err := fsys.OpenRoot("file.txt")
	if err == nil {
		sub.Close()
		t.Fatal("OpenRoot on a regular file should fail")
	}
}

func TestOpenFile_CreateFile(t *testing.T) {
	_, fsys := testHostFS(t)

	flags := int32(oFlagsCreat)
	f, err := fsys.OpenFile("file.txt", false, flags, 0, uint64(RightsFdWrite))
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	defer f.Close()

	if _, err := f.Write([]byte("created")); err != nil {
		t.Fatalf("failed to write: %v", err)
	}
}

func TestOpenFile_CreateExclusive(t *testing.T) {
	_, fsys := testHostFS(t, file("exists.txt", "existing"))

	flags := int32(oFlagsCreat | oFlagsExcl)
	f, err := fsys.OpenFile("exists.txt", false, flags, 0, uint64(RightsFdWrite))
	if err == nil {
		f.Close()
		t.Fatal("expected error for O_CREAT|O_EXCL on existing file")
	}
	if !errors.Is(err, os.ErrExist) {
		t.Errorf("expected os.ErrExist, got %v", err)
	}
}

func TestOpenFile_Truncate(t *testing.T) {
	root, fsys := testHostFS(t, file("trunc.txt", "content"))

	flags := int32(oFlagsTrunc)
	f, err := fsys.OpenFile("trunc.txt", false, flags, 0, uint64(RightsFdWrite))
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	f.Close()

	content, err := os.ReadFile(filepath.Join(root, "trunc.txt"))
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}
	if len(content) != 0 {
		t.Errorf("expected empty file, got %d bytes", len(content))
	}
}

func TestOpenFile_SymlinkNoFollowBlocked(t *testing.T) {
	_, fsys := testHostFS(t, link("escape", "/etc"))

	if _, err := fsys.OpenFile("escape", false, 0, 0, uint64(RightsFdRead)); err == nil {
		t.Fatal("expected error when opening symlink without follow")
	}
}

func TestOpenFile_SymlinkFollowAllowed(t *testing.T) {
	_, fsys := testHostFS(t,
		file("real.txt", "real content"),
		link("link", "real.txt"),
	)

	f, err := fsys.OpenFile("link", true, 0, 0, uint64(RightsFdRead))
	if err != nil {
		t.Fatalf("OpenFile with symlink follow failed: %v", err)
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}
	if string(content) != "real content" {
		t.Errorf("got %q, want %q", string(content), "real content")
	}
}

func TestOpenFile_SymlinkFollowEscapeBlocked(t *testing.T) {
	_, fsys := testHostFS(t, link("escape", "/etc/passwd"))

	f, err := fsys.OpenFile("escape", true, 0, 0, uint64(RightsFdRead))
	if err == nil {
		f.Close()
		t.Fatal("symlink escape succeeded with SYMLINK_FOLLOW")
	}
}

func TestOpenFile_SymlinkWithDotDotInsideSandbox(t *testing.T) {
	_, fsys := testHostFS(t,
		file("f.txt", "parent content"),
		link("subdir/link", "../f.txt"),
	)

	f, err := fsys.OpenFile("subdir/link", true, 0, 0, uint64(RightsFdRead))
	if err != nil {
		t.Fatalf("OpenFile failed for symlink with .. inside sandbox: %v", err)
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}
	if string(content) != "parent content" {
		t.Errorf("got %q, want %q", string(content), "parent content")
	}
}

func TestOpenFile_SymlinkCrossDirectory(t *testing.T) {
	_, fsys := testHostFS(t,
		file("a/b/c/f.txt", "cross dir content"),
		link("a/d/e/link", "../../b/c/f.txt"),
	)

	f, err := fsys.OpenFile("a/d/e/link", true, 0, 0, uint64(RightsFdRead))
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}
	if string(content) != "cross dir content" {
		t.Errorf("got %q, want %q", string(content), "cross dir content")
	}
}

func TestOpenFile_SymlinkWithDotDotEscapeBlocked(t *testing.T) {
	_, fsys := testHostFS(t, link("subdir/link", "../../etc/passwd"))

	if _, err := fsys.OpenFile("subdir/link", true, 0, 0, uint64(RightsFdRead)); err == nil {
		t.Fatal("symlink escape via .. succeeded")
	}
}

func TestOpenFile_IntermediateSymlinkBlocked(t *testing.T) {
	_, fsys := testHostFS(t, link("escape_dir", "/etc"))

	if _, err := fsys.OpenFile("escape_dir/passwd", true, 0, 0, uint64(RightsFdRead)); err == nil {
		t.Fatal("traversing through escaping symlink succeeded")
	}
}

func TestOpenFile_DotDotRejected(t *testing.T) {
	_, fsys := testHostFS(t)

	if _, err := fsys.OpenFile("../etc/passwd", false, 0, 0, uint64(RightsFdRead)); err == nil {
		t.Fatal("expected error for .. escape")
	}
}

func TestOpenFile_AbsolutePathRejected(t *testing.T) {
	_, fsys := testHostFS(t)

	if _, err := fsys.OpenFile("/etc/passwd", false, 0, 0, uint64(RightsFdRead)); err == nil {
		t.Fatal("expected error for absolute path")
	}
}

func TestOpenFile_EmptyPathRejected(t *testing.T) {
	_, fsys := testHostFS(t)

	if _, err := fsys.OpenFile("", false, 0, 0, uint64(RightsFdRead)); err == nil {
		t.Fatal("expected error for empty path")
	}
}

func TestOpenFile_AppendFlag(t *testing.T) {
	root, fsys := testHostFS(t, file("append.txt", "original"))

	flags := int32(fdFlagsAppend)
	f, err := fsys.OpenFile("append.txt", false, 0, flags, uint64(RightsFdWrite))
	if err != nil {
		t.Fatalf("OpenFile failed: %v", err)
	}
	if _, err := f.Write([]byte(" appended")); err != nil {
		t.Fatalf("write failed: %v", err)
	}
	f.Close()

	content, err := os.ReadFile(filepath.Join(root, "append.txt"))
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}
	if string(content) != "original appended" {
		t.Errorf("got %q, want %q", string(content), "original appended")
	}
}

func TestOpenFile_NonExistent(t *testing.T) {
	_, fsys := testHostFS(t)

	_, err := fsys.OpenFile("nonexistent.txt", false, 0, 0, uint64(RightsFdRead))
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected os.ErrNotExist, got %v", err)
	}
}

func TestOpenRoot_CurrentDir(t *testing.T) {
	_, fsys := testHostFS(t)

	sub, err := fsys.OpenRoot(".")
	if err != nil {
		t.Fatalf("OpenRoot . failed: %v", err)
	}
	defer sub.Close()

	info, err := sub.Handle().Stat()
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if !info.IsDir() {
		t.Error("expected directory")
	}
}

func TestOpenFile_IntermediateSymlinkAllowed(t *testing.T) {
	_, fsys := testHostFS(t,
		file("target_dir/file.txt", "through link"),
		link("link", "target_dir"),
	)

	f, err := fsys.OpenFile("link/file.txt", false, 0, 0, uint64(RightsFdRead))
	if err != nil {
		t.Fatalf("OpenFile through intermediate symlink failed: %v", err)
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}
	if string(content) != "through link" {
		t.Errorf("got %q, want %q", string(content), "through link")
	}
}

func TestOpenFile_IntermediateSymlinkWithDotDotAllowed(t *testing.T) {
	_, fsys := testHostFS(t,
		file("target_dir/file.txt", "via dotdot"),
		link("subdir/link", "../target_dir"),
	)

	f, err := fsys.OpenFile("subdir/link/file.txt", false, 0, 0, uint64(RightsFdRead))
	if err != nil {
		t.Fatalf("OpenFile through symlink with .. failed: %v", err)
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}
	if string(content) != "via dotdot" {
		t.Errorf("got %q, want %q", string(content), "via dotdot")
	}
}

func TestOpenFile_ChainedIntermediateSymlinks(t *testing.T) {
	_, fsys := testHostFS(t,
		file("a/b/file.txt", "chained"),
		link("link1", "a"),
		link("link2", "link1/b"),
	)

	f, err := fsys.OpenFile("link2/file.txt", false, 0, 0, uint64(RightsFdRead))
	if err != nil {
		t.Fatalf("OpenFile through chained symlinks failed: %v", err)
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("failed to read: %v", err)
	}
	if string(content) != "chained" {
		t.Errorf("got %q, want %q", string(content), "chained")
	}
}

func TestOpenFile_PermissionBypass_DotDot(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("skipping permission bypass test as root")
	}
	root, fsys := testHostFS(t,
		dir("locked"),
		file("target.txt", "secret"),
	)

	lockedDir := filepath.Join(root, "locked")
	if err := os.Chmod(lockedDir, 0o000); err != nil {
		t.Fatalf("failed to chmod locked dir: %v", err)
	}
	t.Cleanup(func() { os.Chmod(lockedDir, 0o700) })

	// Traversing "locked" requires +x permission. Since it is 000, this should
	// fail.
	f, err := fsys.OpenFile("locked/../target.txt", false, 0, 0, uint64(RightsFdRead))
	if err == nil {
		f.Close()
		t.Fatal("OpenFile bypassed permission check on 'locked' directory")
	}
}

func TestOpenFile_IntermediateFileBlocked(t *testing.T) {
	_, fsys := testHostFS(t, file("notadir", "content"))

	_, err := fsys.OpenFile("notadir/file.txt", false, 0, 0, uint64(RightsFdRead))
	if err == nil {
		t.Fatal("traversing through regular file succeeded")
	}
	if !errors.Is(err, syscall.ENOTDIR) {
		t.Errorf("expected syscall.ENOTDIR, got %v", err)
	}
}

func TestOpenFile_NonExistent_DotDot(t *testing.T) {
	_, fsys := testHostFS(t, file("target.txt", "secret"))

	// "nonexistent/../target.txt" should fail because "nonexistent" does not
	// exist.
	f, err := fsys.OpenFile("nonexistent/../target.txt", false, 0, 0, uint64(RightsFdRead))
	if err == nil {
		f.Close()
		t.Fatal("OpenFile bypassed existence check on 'nonexistent' directory")
	}
}

func TestOpenFile_DotDotEscapeSandboxBlocked(t *testing.T) {
	tmpDir := t.TempDir()
	data := []byte("outside content")
	if err := os.WriteFile(filepath.Join(tmpDir, "outside.txt"), data, 0o644); err != nil {
		t.Fatalf("failed to create outside file: %v", err)
	}

	sandboxRoot := filepath.Join(tmpDir, "sandbox")
	if err := os.MkdirAll(filepath.Join(sandboxRoot, "subdir"), 0o755); err != nil {
		t.Fatalf("failed to create sandbox: %v", err)
	}

	fsys, err := OpenHostFileSystem(sandboxRoot)
	if err != nil {
		t.Fatalf("OpenHostFileSystem failed: %v", err)
	}
	defer fsys.Close()

	// "subdir/../../outside.txt" should fail (escapes sandbox)
	f, err := fsys.OpenFile("subdir/../../outside.txt", false, 0, 0, uint64(RightsFdRead))
	if err == nil {
		f.Close()
		t.Fatal("OpenFile allowed sandbox escape via subdir/../..")
	}
}

func TestOpenFile_TrailingDotDotBlocked(t *testing.T) {
	tmpDir := t.TempDir()
	sandboxRoot := filepath.Join(tmpDir, "sandbox")
	if err := os.Mkdir(sandboxRoot, 0o755); err != nil {
		t.Fatalf("failed to create sandbox root: %v", err)
	}

	fsys, err := OpenHostFileSystem(sandboxRoot)
	if err != nil {
		t.Fatalf("OpenHostFileSystem failed: %v", err)
	}
	defer fsys.Close()

	if _, err := fsys.OpenFile("./..", false, 0, 0, uint64(RightsFdRead)); err == nil {
		t.Fatal("OpenFile allowed escape via ./..")
	}
}

func TestOpenFile_SubdirTrailingDotDotBlocked(t *testing.T) {
	tmpDir := t.TempDir()
	sandboxRoot := filepath.Join(tmpDir, "sandbox")
	if err := os.MkdirAll(filepath.Join(sandboxRoot, "subdir"), 0o755); err != nil {
		t.Fatalf("failed to create sandbox: %v", err)
	}

	fsys, err := OpenHostFileSystem(sandboxRoot)
	if err != nil {
		t.Fatalf("OpenHostFileSystem failed: %v", err)
	}
	defer fsys.Close()

	if _, err := fsys.OpenFile("subdir/../..", false, 0, 0, uint64(RightsFdRead)); err == nil {
		t.Fatal("OpenFile allowed escape via subdir/../..")
	}
}

func TestMkdir_Basic(t *testing.T) {
	root, fsys := testHostFS(t)

	if err := fsys.Mkdir("newdir", 0o755); err != nil {
		t.Fatalf("Mkdir failed: %v", err)
	}

	info, err := os.Stat(filepath.Join(root, "newdir"))
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if !info.IsDir() {
		t.Error("expected directory")
	}
}

func TestMkdir_NestedPath(t *testing.T) {
	root, fsys := testHostFS(t, dir("a"))

	if err := fsys.Mkdir("a/b", 0o755); err != nil {
		t.Fatalf("Mkdir nested failed: %v", err)
	}

	info, err := os.Stat(filepath.Join(root, "a", "b"))
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if !info.IsDir() {
		t.Error("expected directory")
	}
}

func TestMkdir_IntermediateSymlinkBlocked(t *testing.T) {
	escape := t.TempDir()
	_, fsys := testHostFS(t, link("escape", escape))

	if err := fsys.Mkdir("escape/newdir", 0o755); err == nil {
		t.Fatal("Mkdir through escaping symlink should fail")
	}

	if _, statErr := os.Stat(filepath.Join(escape, "newdir")); statErr == nil {
		t.Fatal("directory was created in escape location")
	}
}

func TestMkdir_DotDotRejected(t *testing.T) {
	_, fsys := testHostFS(t)

	if err := fsys.Mkdir("../escape", 0o755); !errors.Is(err, os.ErrInvalid) {
		t.Errorf("expected os.ErrInvalid, got %v", err)
	}
}

func TestMkdir_AbsolutePathRejected(t *testing.T) {
	_, fsys := testHostFS(t)

	if err := fsys.Mkdir("/tmp/escape", 0o755); !errors.Is(err, os.ErrInvalid) {
		t.Errorf("expected os.ErrInvalid, got %v", err)
	}
}

func TestMkdir_EmptyPathRejected(t *testing.T) {
	_, fsys := testHostFS(t)

	if err := fsys.Mkdir("", 0o755); !errors.Is(err, os.ErrInvalid) {
		t.Errorf("expected os.ErrInvalid, got %v", err)
	}
}

func TestMkdir_DotPathRejected(t *testing.T) {
	_, fsys := testHostFS(t)

	if err := fsys.Mkdir(".", 0o755); !errors.Is(err, os.ErrInvalid) {
		t.Errorf("expected os.ErrInvalid for '.', got %v", err)
	}
}

func TestMkdir_ExistingDirectoryFails(t *testing.T) {
	_, fsys := testHostFS(t, dir("existing"))

	if err := fsys.Mkdir("existing", 0o755); !errors.Is(err, os.ErrExist) {
		t.Errorf("expected os.ErrExist, got %v", err)
	}
}

func TestMkdir_ParentNotExistsFails(t *testing.T) {
	_, fsys := testHostFS(t)

	if err := fsys.Mkdir("nonexistent/newdir", 0o755); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected os.ErrNotExist, got %v", err)
	}
}

func TestMkdir_IntermediateSymlinkAllowed(t *testing.T) {
	root, fsys := testHostFS(t,
		dir("target_dir"),
		link("link", "target_dir"),
	)

	if err := fsys.Mkdir("link/newdir", 0o755); err != nil {
		t.Fatalf("Mkdir through intermediate symlink failed: %v", err)
	}

	info, err := os.Stat(filepath.Join(root, "target_dir", "newdir"))
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if !info.IsDir() {
		t.Error("expected directory")
	}
}

func TestStat_BasicFile(t *testing.T) {
	_, fsys := testHostFS(t, file("test.txt", "hello"))

	fs, err := fsys.Stat("test.txt", false)
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if fs.FileType != FileTypeRegularFile {
		t.Errorf("expected regular file, got filetype %d", fs.FileType)
	}
	if fs.Size != 5 {
		t.Errorf("expected size 5, got %d", fs.Size)
	}
}

func TestStat_Directory(t *testing.T) {
	_, fsys := testHostFS(t, dir("subdir"))

	fs, err := fsys.Stat("subdir", false)
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if fs.FileType != FileTypeDirectory {
		t.Errorf("expected directory, got filetype %d", fs.FileType)
	}
}

func TestStat_CurrentDir(t *testing.T) {
	_, fsys := testHostFS(t)

	fs, err := fsys.Stat(".", false)
	if err != nil {
		t.Fatalf("Stat . failed: %v", err)
	}
	if fs.FileType != FileTypeDirectory {
		t.Errorf("expected directory, got filetype %d", fs.FileType)
	}
}

func TestStat_NestedPath(t *testing.T) {
	_, fsys := testHostFS(t, file("a/b/test.txt", "nested"))

	fs, err := fsys.Stat("a/b/test.txt", false)
	if err != nil {
		t.Fatalf("Stat nested failed: %v", err)
	}
	if fs.FileType != FileTypeRegularFile {
		t.Errorf("expected regular file, got filetype %d", fs.FileType)
	}
}

func TestStat_SymlinkNoFollow(t *testing.T) {
	_, fsys := testHostFS(t,
		file("real.txt", "content"),
		link("link", "real.txt"),
	)

	fs, err := fsys.Stat("link", false)
	if err != nil {
		t.Fatalf("Stat symlink failed: %v", err)
	}
	if fs.FileType != FileTypeSymbolicLink {
		t.Errorf("expected symbolic link, got filetype %d", fs.FileType)
	}
}

func TestStat_SymlinkFollow(t *testing.T) {
	_, fsys := testHostFS(t,
		file("real.txt", "content"),
		link("link", "real.txt"),
	)

	fs, err := fsys.Stat("link", true)
	if err != nil {
		t.Fatalf("Stat symlink with follow failed: %v", err)
	}
	if fs.FileType != FileTypeRegularFile {
		t.Errorf("expected regular file, got filetype %d", fs.FileType)
	}
	if fs.Size != 7 {
		t.Errorf("expected size 7, got %d", fs.Size)
	}
}

func TestStat_IntermediateSymlinkBlocked(t *testing.T) {
	escape := t.TempDir()
	data := []byte("secret")
	if err := os.WriteFile(filepath.Join(escape, "secret.txt"), data, 0o644); err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	_, fsys := testHostFS(t, link("escape", escape))

	if _, err := fsys.Stat("escape/secret.txt", false); err == nil {
		t.Fatal("Stat through escaping symlink should fail")
	}
}

func TestStat_DotDotRejected(t *testing.T) {
	_, fsys := testHostFS(t)

	if _, err := fsys.Stat("../etc/passwd", false); err == nil {
		t.Fatal("expected error for .. escape")
	}
}

func TestStat_AbsolutePathRejected(t *testing.T) {
	_, fsys := testHostFS(t)

	if _, err := fsys.Stat("/etc/passwd", false); err == nil {
		t.Fatal("expected error for absolute path")
	}
}

func TestStat_NonExistent(t *testing.T) {
	_, fsys := testHostFS(t)

	if _, err := fsys.Stat("nonexistent.txt", false); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected os.ErrNotExist, got %v", err)
	}
}

func TestStat_SymlinkFollowEscapeBlocked(t *testing.T) {
	_, fsys := testHostFS(t, link("escape", "/etc/passwd"))

	if _, err := fsys.Stat("escape", true); err == nil {
		t.Fatal("Stat with symlink follow should fail for escape symlink")
	}
}

func TestStat_SymlinkChainWithDotDotEscapeBlocked(t *testing.T) {
	_, fsys := testHostFS(t, link("subdir/link", "../../etc/passwd"))

	if _, err := fsys.Stat("subdir/link", true); err == nil {
		t.Fatal("Stat with symlink follow should fail for .. escape")
	}
}

func TestStat_IntermediateSymlinkAllowed(t *testing.T) {
	_, fsys := testHostFS(t,
		file("target_dir/file.txt", "stat through link"),
		link("link", "target_dir"),
	)

	fs, err := fsys.Stat("link/file.txt", false)
	if err != nil {
		t.Fatalf("Stat through intermediate symlink failed: %v", err)
	}
	if fs.FileType != FileTypeRegularFile {
		t.Errorf("expected regular file, got filetype %d", fs.FileType)
	}
}

func TestChtimes_BasicFile(t *testing.T) {
	root, fsys := testHostFS(t, file("test.txt", "hello"))
	atime := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).UnixNano()
	mtime := time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC).UnixNano()

	fstFlags := fstFlagsAtim | fstFlagsMtim
	if err := fsys.Chtimes("test.txt", atime, mtime, fstFlags, false); err != nil {
		t.Fatalf("Chtimes failed: %v", err)
	}

	info, err := os.Stat(filepath.Join(root, "test.txt"))
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	// Note: AccessTime check depends on FS support so we only verify ModTime.
	if info.ModTime().UnixNano() != mtime {
		t.Errorf("file Mtime: got %v, want %v", info.ModTime(), mtime)
	}
}

func TestChtimes_SymlinkFollow(t *testing.T) {
	root, fsys := testHostFS(t,
		file("target.txt", "content"),
		link("link", "target.txt"),
	)
	mtime := time.Date(2022, 1, 1, 0, 0, 0, 0, time.UTC).UnixNano()

	if err := fsys.Chtimes("link", 0, mtime, fstFlagsMtim, true); err != nil {
		t.Fatalf("Chtimes with followSymlinks=true failed: %v", err)
	}

	targetInfo, err := os.Stat(filepath.Join(root, "target.txt"))
	if err != nil {
		t.Fatalf("Stat target failed: %v", err)
	}
	if targetInfo.ModTime().UnixNano() != mtime {
		t.Errorf("target Mtime: got %v, want %v", targetInfo.ModTime(), mtime)
	}
}

func TestChtimes_SymlinkNoFollow(t *testing.T) {
	root, fsys := testHostFS(t,
		file("target.txt", "content"),
		link("link", "target.txt"),
	)
	targetBefore, err := os.Stat(filepath.Join(root, "target.txt"))
	if err != nil {
		t.Fatalf("Stat target failed: %v", err)
	}
	mtime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC).UnixNano()

	// No follow: should update the symlink itself, NOT the target.
	if err := fsys.Chtimes("link", 0, mtime, fstFlagsMtim, false); err != nil {
		t.Fatalf("Chtimes with followSymlinks=false failed: %v", err)
	}

	targetAfter, err := os.Stat(filepath.Join(root, "target.txt"))
	if err != nil {
		t.Fatalf("Stat target failed: %v", err)
	}
	if !targetAfter.ModTime().Equal(targetBefore.ModTime()) {
		t.Errorf("target Mtime changed")
	}

	linkInfo, err := os.Lstat(filepath.Join(root, "link"))
	if err != nil {
		t.Fatalf("Lstat link failed: %v", err)
	}
	if linkInfo.ModTime().UnixNano() != mtime {
		t.Errorf("symlink Mtime: got %v, want %v", linkInfo.ModTime(), mtime)
	}
}

func TestChtimes_Directory(t *testing.T) {
	root, fsys := testHostFS(t, dir("subdir"))
	mtime := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)

	if err := fsys.Chtimes("subdir", 0, mtime.UnixNano(), fstFlagsMtim, false); err != nil {
		t.Fatalf("Chtimes on directory failed: %v", err)
	}

	info, err := os.Stat(filepath.Join(root, "subdir"))
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if info.ModTime().Sub(mtime).Abs() > time.Second {
		t.Errorf("dir Mtime: got %v, want %v", info.ModTime(), mtime)
	}
}

func TestLink_Basic(t *testing.T) {
	root, fsys := testHostFS(t, file("original.txt", "content"))

	if err := fsys.Link("original.txt", false, fsys, "hardlink.txt"); err != nil {
		t.Fatalf("Link failed: %v", err)
	}

	content, err := os.ReadFile(filepath.Join(root, "hardlink.txt"))
	if err != nil {
		t.Fatalf("failed to read hard link: %v", err)
	}
	if string(content) != "content" {
		t.Errorf("got %q, want %q", string(content), "content")
	}

	origInfo, _ := os.Stat(filepath.Join(root, "original.txt"))
	linkInfo, _ := os.Stat(filepath.Join(root, "hardlink.txt"))
	origStat := origInfo.Sys().(*syscall.Stat_t)
	linkStat := linkInfo.Sys().(*syscall.Stat_t)
	if origStat.Ino != linkStat.Ino {
		t.Errorf("inodes differ: %d vs %d", origStat.Ino, linkStat.Ino)
	}
}

func TestLink_SymlinkEscapeBlocked(t *testing.T) {
	escape := t.TempDir()
	targetFile := filepath.Join(escape, "target.txt")
	if err := os.WriteFile(targetFile, []byte("secret"), 0o644); err != nil {
		t.Fatalf("failed to create target: %v", err)
	}

	_, fsys := testHostFS(t, link("escape", escape))

	// Try to create a hard link through the escape symlink.
	if err := fsys.Link("escape/target.txt", true, fsys, "stolen.txt"); err == nil {
		t.Fatal("Link through escape symlink should fail")
	}
}

func TestReadlink_Basic(t *testing.T) {
	_, fsys := testHostFS(t, link("mylink", "target.txt"))

	target, err := fsys.Readlink("mylink")
	if err != nil {
		t.Fatalf("Readlink failed: %v", err)
	}
	if target != "target.txt" {
		t.Errorf("got %q, want %q", target, "target.txt")
	}
}

func TestReadlink_NonSymlink(t *testing.T) {
	_, fsys := testHostFS(t, file("regular.txt", "content"))

	if _, err := fsys.Readlink("regular.txt"); err == nil {
		t.Fatal("Readlink on regular file should fail")
	}
}

func TestRmdir_Basic(t *testing.T) {
	root, fsys := testHostFS(t, dir("emptydir"))

	if err := fsys.Rmdir("emptydir"); err != nil {
		t.Fatalf("Rmdir failed: %v", err)
	}

	if _, statErr := os.Stat(filepath.Join(root, "emptydir")); !os.IsNotExist(statErr) {
		t.Error("directory should have been removed")
	}
}

func TestRmdir_NotEmpty(t *testing.T) {
	_, fsys := testHostFS(t, file("notempty/file.txt", "content"))

	if err := fsys.Rmdir("notempty"); err == nil {
		t.Fatal("Rmdir on non-empty directory should fail")
	}
}

func TestRmdir_OnFileFails(t *testing.T) {
	_, fsys := testHostFS(t, file("file.txt", "content"))

	if err := fsys.Rmdir("file.txt"); err == nil {
		t.Fatal("Rmdir on a regular file should fail")
	}
}

func TestRename_Basic(t *testing.T) {
	root, fsys := testHostFS(t, file("old.txt", "content"))

	if err := fsys.Rename(fsys, "old.txt", "new.txt"); err != nil {
		t.Fatalf("Rename failed: %v", err)
	}

	if _, statErr := os.Stat(filepath.Join(root, "old.txt")); !os.IsNotExist(statErr) {
		t.Error("old file should not exist")
	}

	content, err := os.ReadFile(filepath.Join(root, "new.txt"))
	if err != nil {
		t.Fatalf("failed to read new file: %v", err)
	}
	if string(content) != "content" {
		t.Errorf("got %q, want %q", string(content), "content")
	}
}

func TestSymlink_Basic(t *testing.T) {
	root, fsys := testHostFS(t, file("target.txt", "content"))

	if err := fsys.Symlink("target.txt", "newlink"); err != nil {
		t.Fatalf("Symlink failed: %v", err)
	}

	linkTarget, err := os.Readlink(filepath.Join(root, "newlink"))
	if err != nil {
		t.Fatalf("Readlink failed: %v", err)
	}
	if linkTarget != "target.txt" {
		t.Errorf("got %q, want %q", linkTarget, "target.txt")
	}
}

func TestUnlink_Basic(t *testing.T) {
	root, fsys := testHostFS(t, file("file.txt", "content"))

	if err := fsys.Unlink("file.txt"); err != nil {
		t.Fatalf("Unlink failed: %v", err)
	}

	if _, statErr := os.Stat(filepath.Join(root, "file.txt")); !os.IsNotExist(statErr) {
		t.Error("file should have been deleted")
	}
}

func TestUnlink_Directory(t *testing.T) {
	_, fsys := testHostFS(t, dir("mydir"))

	if err := fsys.Unlink("mydir"); err == nil {
		t.Fatal("Unlink on directory should fail")
	}
}
