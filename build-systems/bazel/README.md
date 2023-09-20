Using Bazel's apt repository

Step 1: Add Bazel distribution URI as a package source
```
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
```

Step 2: Install and update Bazel

```
sudo apt update && sudo apt install baze
```


# Examples to build C++ code

This package will showcase how to build C++ code in stages.

### Stage 1
The first stage is really simple and shows you how to compile a binary with a single source file.

### Stage 2
The second stage will showcase how to build an application with multiple source and header files, separated in a library and a binary.

### Stage 3
The third stage showcases how to link multiple build directories by building multiple libraries in different packages and then connecting it up with the main application.