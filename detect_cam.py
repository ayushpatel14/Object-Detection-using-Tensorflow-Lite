from time import time
t1=time()
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

t2=time()
print("Libraries loaded in ",t2-t1)
cap=cv2.VideoCapture(0)		#Start camera instance.0 denotes the webcam (if present) else the first cam atteched
interpreter = Interpreter(model_path="detect.tflite")	#loading the model 
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()     #to help find out input format to be given(to model)
output_details = interpreter.get_output_details()   #to help find out output format of model
floating_model = input_details[0]['dtype'] == np.float32    #to check model type i.e Quantized model or Floating Point model
#print(input_details)   
#print(output_details)  
t1=time()
print("Initialized in ",t1-t2)

while (True):
    
    t1=time()
    ret, frame=cap.read()   # reading frames from cam
    # Image optimisation
    #h,w=frame.shape[:2]
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    frameo=cv2.resize(frame,(height,width)) #resizing frame according to model prescribed size 
    
    if floating_model:
        input_data = (np.float32(input_data)/218) -1
    else:
        input_data=np.expand_dims(frameo,axis=0)    # Expand the shape of array
    
    interpreter.set_tensor(input_details[0]['index'], input_data)   #passing data to model
    interpreter.invoke()        #making predictions
    
    boundbox = interpreter.get_tensor(output_details[0]['index'])   #getting output
    obj_class = interpreter.get_tensor(output_details[1]['index'])  #getting outout
    score = interpreter.get_tensor(output_details[2]['index'])      #getting output
    num = interpreter.get_tensor(output_details[3]['index']) #Always equals to 10
   
    for i in range(int(num)):
        
        top, left, bottom, right = boundbox[0][i]   #getting the postion of detected object
        classId = int(obj_class[0][i])              #getting class of object
        scores = score[0][i]                        #getting predction score of that object
        
        if scores > 0.5:                            #if score > 50%
            x1 =int( left * width)                  
            y1 =int( bottom * height)
            x2 =int( right * width)
            y2 =int(top * height)
            labels=load_labels("labelmap.txt")
            if labels:
                  print(labels[classId], 'score = ', scores)
            else:
                  print ('score = ', scores)
            #draw_rectangle(frame, box, (128,128,20), width=5)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
            if labels:
                 cv2.putText( frame,labels[classId], (x1+20,y1+20), cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,255))
    
    cv2.imshow("Detected",frame)    
    t2=time()
    print("Time Taken: ",t2-t1)
    if cv2.waitKey(100)==13:    #press enter to abort
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()
