class CameraIntrinsicParameter:
    def __init__(self, f:float, cx:float, cy:float, width:float, height:float) -> None:
        self.f = f
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
