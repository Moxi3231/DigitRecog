class BBoxUtil:
    def __init__(self, iou_threshold:float = 0.1, width_threshold:float = 0.3, length_threshold:float = 0.3) -> None:
        self.iou_threshild = iou_threshold
        self.width_threshold = width_threshold
        self.length_threshold = length_threshold

    @staticmethod
    def calculate_iou(box1,box2) -> float:
        top_left_crns = (max(box1[0],box2[0]),max(box1[1],box2[1]))
        btm_rgth_crns = (min(box1[0]+box1[2],box2[0]+box2[2]),min(box1[1]+box1[3],box2[1]+box2[3]))

        w,l = max(0,btm_rgth_crns[0] - top_left_crns[0]),max(0,btm_rgth_crns[1]-top_left_crns[1])
        intersect_area = w*l
        union_area = (box1[2]*box1[3]) + (box2[2]*box2[3]) - intersect_area
        iou = intersect_area/max(union_area,1)
        return iou

    """
    boxes: Bounding boxes in form of minx,miny, width, length
    """
    def apply_nms(self,boxes):
        
        return