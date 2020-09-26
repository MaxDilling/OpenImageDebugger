# -*- coding: utf-8 -*-

"""
Module developed for quick testing the OpenImageDebugger shared library
"""

# From OpenCV
CV_CN_MAX = 512
CV_CN_SHIFT = 3
CV_DEPTH_MAX = (1 << CV_CN_SHIFT)
CV_MAT_DEPTH_MASK = (CV_DEPTH_MAX - 1)
CV_MAT_DEPTH = lambda flags: ((flags) & CV_MAT_DEPTH_MASK)
CV_MAT_CN_MASK = ((CV_CN_MAX - 1) << CV_CN_SHIFT)
CV_MAT_CN = lambda flags: ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
CV_ELEM_SIZE1 = lambda type: ((0x28442211 >> CV_MAT_DEPTH(type)*4) & 15)
CV_ELEM_SIZE = lambda type: (CV_MAT_CN(type)*CV_ELEM_SIZE1(type))

import sys
import os
import math
import time
import threading
import array
import cv2
import numpy as np

from oidscripts import oidwindow
from oidscripts import symbols
from oidscripts.debuggers.interfaces import BridgeInterface

def oidshowfile(script_path, files):
    """
    Entry point for the testing mode.
    """

    dummy_debugger = DummyDebugger(files)

    window = oidwindow.OpenImageDebuggerWindow(script_path, dummy_debugger)
    window.initialize_window()

    try:
        # Wait for window to initialize
        while not window.is_ready():
            time.sleep(0.1)

        window.set_available_symbols(dummy_debugger.get_available_symbols())

        for buffer in dummy_debugger.get_available_symbols():
            window.plot_variable(buffer)

        while window.is_ready():
            dummy_debugger.run_event_loop()
            time.sleep(0.1)

    except KeyboardInterrupt:
        window.terminate()
        exit(0)

    dummy_debugger.kill()


def _mat_to_buffer(mat, filename):
    data = mat.tobytes()
    channels = 1 if len(mat.shape) == 2 else mat.shape[2]
    types = [np.uint8, np.uint8, np.uint16, np.int16, np.int32, np.float32]
    
    if mat.dtype not in types:
        print ("Error: Type %s is not supported. Skip '%s'" % (mat.dtype, filename))
        return None
        
    return {
        'variable_name': filename,
        'display_name': str(mat.dtype) + '* ' + os.path.basename(filename),
        'pointer': memoryview(data),
        'width': int(mat.shape[1]),
        'height': int(mat.shape[0]),
        'channels': channels,
        'type': types.index(mat.dtype),
        'row_stride': int(mat.shape[1]),
        'pixel_layout': 'rgba',
        'transpose_buffer': False
    }

def _read_mat(filename):
    file = open(filename, "rb")

    hr = file.readline()

    if hr == b'TYPE BIN\n':
        rows = int.from_bytes(file.read(4), 'little')
        cols = int.from_bytes(file.read(4), 'little')
        type = int.from_bytes(file.read(4), 'little')
        channelsT = int.from_bytes(file.read(4), 'little')
        data = file.read(CV_ELEM_SIZE(type) * rows * cols)

        if sys.version_info[0] == 2:
            mem1 = buffer(data)
        else:
            mem1 = memoryview(data)
            
        return {
            'variable_name': filename,
            'display_name': 'float* ' + os.path.basename(filename),
            'pointer': mem1,
            'width': cols,
            'height': rows,
            'channels': channelsT,
            'type': CV_MAT_DEPTH(type),
            'row_stride': cols,
            'pixel_layout': 'rgba',
            'transpose_buffer': False
        }
            
    elif hr == b'TYPE HR\n':
        fs = cv2.FileStorage(file.read().decode(), cv2.FileStorage_MEMORY)
        return _mat_to_buffer(fs.getNode("mat").mat(), filename)
        
    else:
        print("Can't read file " + filename)
        return None
        

def _read_img(filename):
    img = cv2.imread(filename)
    return _mat_to_buffer(img, filename)
    
def _read_buffers(filenames):
    buffers = {}
    
    for filename in filenames:
        if os.path.isfile(filename):
            buffer = _read_mat(filename) if filename[-4:] in [".mat", ".bin"] else _read_img(filename)
            if buffer:
                buffers[filename] = buffer
        
        else:
            print ("Error: File '%s' does not exist" % filename)
        
    return buffers


class DummyDebugger(BridgeInterface):
    """
    Very simple implementation of a debugger bridge for the sake of the test
    mode.
    """
    def __init__(self, files):
        self._buffers = _read_buffers(files)
        self._buffer_names = [name for name in self._buffers]

        self._is_running = True
        self._incoming_request_queue = []

    def run_event_loop(self):
        if self._is_running:
            request_queue = self._incoming_request_queue
            self._incoming_request_queue = []

            while len(request_queue) > 0:
                latest_request = request_queue.pop(-1)
                latest_request()

    def kill(self):
        """
        Request consumer thread to finish its execution
        """
        self._is_running = False

    def get_casted_pointer(self, typename, debugger_object):
        """
        No need to cast anything in this example
        """
        return debugger_object

    def register_event_handlers(self, events):
        """
        No need to register events in this example
        """
        pass

    def get_available_symbols(self):
        """
        Return the names of the available sample buffers
        """
        return self._buffer_names

    def get_buffer_metadata(self, var_name):
        """
        Search in the list of available buffers and return the requested one
        """
        if var_name in self._buffers:
            return self._buffers[var_name]

        return None

    def get_backend_name(self):  # type: () -> str
        return 'dummy'

    def queue_request(self, callable_request):
        self._incoming_request_queue.append(callable_request)
