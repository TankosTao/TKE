from .detector_decode import Decode4


def build_decode(decode_name,**kwargs):
    head = eval(decode_name)(**kwargs)
    return head
