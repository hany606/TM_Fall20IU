import cv2
import numpy as np
import sys

for exp_i in range(5):
  for exp_j in range(3):
    # exp_i = 2-1
    # exp_j = 2-1
    video_name = f"Exp{exp_i+1}{exp_j+1}.mp4"


    # video_name = "Exp41.mp4"
    # if(len(sys.argv) >= 2):
      # video_name = sys.argv[1]


    print(f"Openning Video: {video_name}")
    cap = cv2.VideoCapture(video_name)

    def rescale_frame(frame, percent=75):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


    frame_prev = np.zeros(shape=(480,100))
    steps = 0
    yoyo_y_px = []
    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      
      if ret == True:
        # print(frame.shape)

        frame_original = cv2.rotate(rescale_frame(frame, percent=25), cv2.ROTATE_90_CLOCKWISE)[10:,100:200]
        # print(frame.shape)

        frame_gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if(len(contours) > 0):
          c = max(contours, key = cv2.contourArea)
          x,y,w,h = cv2.boundingRect(c)

        frame_rec = cv2.rectangle(frame_original,(x,y),(x+w,y+h),(255,0,0),1)
        
        yoyo_y = y+h/2
        yoyo_y_px.append(yoyo_y)
        print(f"Y = {yoyo_y} in pixels, Y = {yoyo_y*69/383.5} in cm (approx),  at timesteps: {steps}")
        # frame_rgb_blank = np.zeros((frame.shape[0], frame.shape[1],3))
        # frame_rgb_blank[:,:,] = frame
        frame_out = cv2.circle(frame_rec, (x+w//2,y+h//2), 3, [0,255,0], -1)

        new_frame = frame_out
        # new_frame = frame - frame_prev
        
        cv2.imshow('Video', new_frame)
        frame_prev = frame
        steps += 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break



      # Break the loop
      else: 
        break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


    # print(np.mean(yoyo_y_px[-5:-1]), yoyo_y_px[-1])
    # Accuracte
    string_length = [69, 49, 42, 35, 26]
    yoyo_y_cm = [i*string_length[int(video_name[3])-1]/np.mean(yoyo_y_px[-5:-1]) for i in yoyo_y_px]

    # for i in yoyo_y_cm:
    #   print(i)

    np.save(f"Exp{exp_i+1}{exp_j+1}_cm.npy", yoyo_y_cm)