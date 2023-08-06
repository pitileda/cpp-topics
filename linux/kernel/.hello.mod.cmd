cmd_/home/ihor/projects/kernel/hello.mod := printf '%s\n'   hello.o | awk '!x[$$0]++ { print("/home/ihor/projects/kernel/"$$0) }' > /home/ihor/projects/kernel/hello.mod
