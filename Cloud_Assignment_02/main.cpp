#include <iostream>
#include <unistd.h>
#include <sched.h>
#include <sys/mount.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>

#define STACK_SIZE (1024 * 1024) // Stack size for child process

char child_stack[STACK_SIZE];

int child_process(void* args) {
    // Set hostname
    sethostname("my_container", strlen("my_container"));

    // Mount a new proc filesystem
    mount("proc", "/proc", "proc", 0, "");

    // Change root filesystem
    if (chroot("/my_rootfs") == -1) {
        perror("chroot");
        return 1;
    }
    chdir("/");

    // Start the shell
    execlp("/bin/sh", "/bin/sh", nullptr);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <rootfs_path>\n";
        return 1;
    }

    const char* rootfs_path = argv[1];

    // Create namespaces using clone
    pid_t child_pid = clone(child_process, child_stack + STACK_SIZE,
                            CLONE_NEWUTS | CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWNET | SIGCHLD, nullptr);
    if (child_pid == -1) {
        perror("clone");
        return 1;
    }

    waitpid(child_pid, nullptr, 0); // Wait for child to finish
    return 0;
}
