import cv2 as cv
import numpy as np




def composite(background,shadow,model,mask):


    background=background/255.0
    if model.dtype=="uint8":
        model=model/255.0
    else:
        model=model/65535.0
    shadow=shadow/255.0


    alpha=model[:,:,2]
    alpha=np.where(alpha>0.8,0,0.7)
    alpha = np.where(mask > alpha, mask, alpha)
    alpha=alpha[:,:,np.newaxis]
    alpha=alpha.repeat(3,axis=2)

    model=model[:,:,0:3]

    ones=np.ones(alpha.shape,dtype=float)
    shadow=ones-shadow*(1.0-0.7)
    shadow=cv.GaussianBlur(shadow,(9,9),1.5)

    shadow=model*shadow
    final=shadow*alpha+(ones-alpha)*background

    # final=model*alpha+bg*(ones-alpha)
    return final*255.0

if __name__=="__main__":

    bg=cv.imread("./input/COCO_train2014_000000002444_background.jpg")
    model=cv.imread("./input/COCO_train2014_000000002444_newshadow.jpg",-1)
    shadow=cv.imread("./input/COCO_train2014_000000002444_premask.png")
    mask=cv.imread("./input/COCO_train2014_000000002444_mask.exr", 0)
    final=composite(bg,shadow,model,mask)
    cv.imwrite("./result/COCO_train2014_000000002444.png",final)




