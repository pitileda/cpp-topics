cmd_/home/ihor/projects/kernel/modules.order := {   echo /home/ihor/projects/kernel/hello.ko; :; } | awk '!x[$$0]++' - > /home/ihor/projects/kernel/modules.order
