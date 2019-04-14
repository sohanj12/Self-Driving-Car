import win32api as wapi
import time

keylist = ['\b']
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keylist.append(char)

def key_check():
    keys = []
    for key in keylist:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
