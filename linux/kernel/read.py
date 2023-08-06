#!/usr/bin/env python
import os

dev = os.open("/dev/lkm_example", os.O_RDWR)
print(os.read(dev, 16))
