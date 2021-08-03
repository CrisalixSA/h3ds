#!/bin/bash

yapf -rip h3ds/*.py --style google
yapf -rip examples/*.py --style google
yapf -rip tests/*.py --style google