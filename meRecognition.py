import cv2
import torchvision.transforms as transforms
import torch
import MER

cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture("http://192.168.43.1:4747/video")# connecting to ip cam
# cam = cv2.VideoCapture('./FER/realTime/Videos/all.mp4')# video
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
#print(width,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# GPU if available, else CPU
device = MER.get_default_device()
print("Selected device:",device)

# Loading pretrained weights
w = '.\FER\\realTime\MERCnn.pth'
model = MER.to_device(MER.MERCnnModel(),device)
if str(device) == 'cpu':
    model.load_state_dict(torch.load(w,map_location=torch.device('cpu'))) # use for cpu
if str(device) == 'gpu':
    model.load_state_dict(torch.load(w,map_location=torch.device('cuda'))) #for GPU

while True:
    _ , frame = cam.read()
    if _:
        bBox = MER.faceBox(frame)
    bBox = MER.faceBox(frame)
    if len(bBox) > 0:
        for box in bBox:
            x,y,w,h = int(box.xmin*width),int(box.ymin*height),int(box.width*width),int(box.height*height)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            faceExp = frame[y:y+h,x:x+w]
            try:
                faceExpResized = cv2.resize(faceExp,(80,80))
                transform = transforms.ToTensor()
                faceExpResizedTensor = transform(faceExpResized)
            except:
                continue
            prediction = MER.predict_image(faceExpResizedTensor, model, device)
            cv2.putText(frame,prediction,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))

    cv2.imshow('MER', frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'): # to quit the camera press 'q'
        print('end')
        break
cam.release()
cv2.destroyAllWindows()