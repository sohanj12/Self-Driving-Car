import numpy as np
from PIL import ImageGrab
import cv2
import win32gui
import win32ui, win32con, win32api

#cv2.namedWindow('window', cv2.WINDOW_KEEPRATIO)
'''while(True):
    printscreen_pil = ImageGrab.grab(bbox = (0, 80, 800, 640))
    printscreen_numpy = np.array(printscreen_pil.getdata(), dtype = 'uint8')\
    .reshape((printscreen_pil.size[1], printscreen_pil.size[0], 3))
    cv2.imshow('window', printscreen_numpy)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break'''

def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()
    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

'''while True:
    curr_view = grab_screen([0,30,800,600])
    cv2.imshow('frame', curr_view)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
'''
