import os
import glob
import jsonpickle
import cv2
import math
import random as rnd

#global Neck_1
#global MidHip_8
#from keypoints import People
#from keypoints import KeyPoints
#from keypoints import angleT


class IdGen():
    def __init__(self):
        self.counter = -1
        
    def getid(self):
        self.counter += 1
        return self.counter

class Pose2d():
    def __init__(self):
        
        self.Nose_0 = ()
        self.Neck_1 = ()
        self.RShoulder_2 = ()
        self.RElbow_3 = ()
        self.RWrist_4 = ()
        self.LShoulder_5 = ()
        self.LElbow_6 = ()
        self.LWrist_7 = ()
        self.MidHip_8 = ()
        self.RHip_9 = ()
        self.RKnee_10 = ()
        self.RAnkle_11 = ()
        self.LHip_12 = ()
        self.LKnee_13 = ()
        self.LAnkle_14 = ()
        self.REye_15 = ()
        self.LEye_16 = ()
        self.REar_17 = ()
        self.LEar_18 = ()
        self.LBigToe_19 = ()
        self.LSmallToe_20 = ()
        self.LHeel_21 = ()
        self.RBigToe_22 = ()
        self.RSmallToe_23 = ()
        self.RHeel_24 = ()

class People():
    
    def __init__(self, person_id, posekeypoints2d):
        
        pose = Pose2d()
        pose.Nose_0 = ((posekeypoints2d[0], posekeypoints2d[1]), posekeypoints2d[2])
        pose.Neck_1 = ((posekeypoints2d[3], posekeypoints2d[4]), posekeypoints2d[5])
        global Neck_1
        Neck_1 = pose.Neck_1
        pose.RShoulder_2 = ((posekeypoints2d[6], posekeypoints2d[7]), posekeypoints2d[8])
        pose.RElbow_3 = ((posekeypoints2d[9], posekeypoints2d[10]), posekeypoints2d[11])
        pose.RWrist_4 = ((posekeypoints2d[12], posekeypoints2d[13]), posekeypoints2d[14])
        pose.LShoulder_5 = ((posekeypoints2d[15], posekeypoints2d[16]), posekeypoints2d[17])
        pose.LElbow_6 = ((posekeypoints2d[18], posekeypoints2d[19]), posekeypoints2d[20])
        pose.LWrist_7 = ((posekeypoints2d[21], posekeypoints2d[22]), posekeypoints2d[23])
        pose.MidHip_8 = ((posekeypoints2d[24], posekeypoints2d[25]), posekeypoints2d[26])
        global MidHip_8
        MidHip_8 = pose.MidHip_8
        pose.RHip_9 = ((posekeypoints2d[27], posekeypoints2d[28]), posekeypoints2d[29])
        pose.RKnee_10 = ((posekeypoints2d[30], posekeypoints2d[31]), posekeypoints2d[32])
        pose.RAnkle_11 = ((posekeypoints2d[33], posekeypoints2d[34]), posekeypoints2d[35])
        pose.LHip_12 = ((posekeypoints2d[36], posekeypoints2d[37]), posekeypoints2d[38])
        pose.LKnee_13 = ((posekeypoints2d[39], posekeypoints2d[40]), posekeypoints2d[41])
        pose.LAnkle_14 = ((posekeypoints2d[42], posekeypoints2d[43]), posekeypoints2d[44])
        pose.REye_15 = ((posekeypoints2d[45], posekeypoints2d[46]), posekeypoints2d[47])
        pose.LEye_16 = ((posekeypoints2d[48], posekeypoints2d[49]), posekeypoints2d[50])
        pose.REar_17 = ((posekeypoints2d[51], posekeypoints2d[52]), posekeypoints2d[53])
        pose.LEar_18 = ((posekeypoints2d[54], posekeypoints2d[55]), posekeypoints2d[56])
        pose.LBigToe_19 = ((posekeypoints2d[57], posekeypoints2d[58]), posekeypoints2d[59])
        pose.LSmallToe_20 = ((posekeypoints2d[60], posekeypoints2d[61]), posekeypoints2d[62])
        pose.LHeel_21 = ((posekeypoints2d[63], posekeypoints2d[64]), posekeypoints2d[65])
        pose.RBigToe_22 = ((posekeypoints2d[66], posekeypoints2d[67]), posekeypoints2d[68])
        pose.RSmallToe_23 = ((posekeypoints2d[69], posekeypoints2d[70]), posekeypoints2d[71])
        pose.RHeel_24 = ((posekeypoints2d[72], posekeypoints2d[73]), posekeypoints2d[74])
        
        self.pose2d = pose
        self.person_id = person_id
        self.color = (rnd.randrange(0, 255), rnd.randrange(0, 255), rnd.randrange(0, 255))
        
        self.stable = False
        self.traced = False
        self.tracedcount = 0
        
class KeyPoints():
    def __init__(self, sourcedict):
        self.peoplelist = self.extractpeople(sourcedict['people'])
        self.version = sourcedict['version']
        self.idgen = IdGen()
        
        self.confidencethreshold = 0.5
        self.circleradius = 3
        self.linewidth = 2
        
        self.movethreshold = 50
        self.stablethreshold = 2
        
        
    def extractpeople(self, peopledict):
        
        peoplelist = []
        
        for peopled in peopledict:
            person_id = peopled['person_id'][0]
            pose_keypoints_2d = peopled['pose_keypoints_2d']    
            p = People(person_id, pose_keypoints_2d)
            peoplelist.append(p)
            
        return peoplelist
            
    def initid(self):
        for people in self.peoplelist:
            people.person_id = self.idgen.getid()
        return self
    
    def traceid(self, fkp):
        
        for kppeople in fkp.peoplelist:
        
            ((kph8, kpw8), kpc8) = kppeople.pose2d.MidHip_8
            
            if (kpc8 > self.confidencethreshold): 
                
                candidate = None 
                dist = 999999999            

                for people in self.peoplelist: 
                    
                    ((h8, w8), c8) = people.pose2d.MidHip_8
                    
                    if (c8 > self.confidencethreshold):
                        
                        tmpdist = math.hypot(h8 - kph8, w8 - kpw8)
                        
                        if (dist > tmpdist):
                            if (tmpdist < self.movethreshold):
                                dist = tmpdist
                                candidate = people
                            else:
                                print('dist = {}'.format(tmpdist))

                                
                if not (candidate is None):
                    
                    candidate.pose2d = kppeople.pose2d
                    candidate.stable = (dist < self.stablethreshold)
                    candidate.traced = True
                    candidate.tracedcount += 1
                    
                else:
                    kppeople.person_id = self.idgen.getid()
                    self.peoplelist.append(kppeople)
                
            
            
        
        return self

    def drawsk(self, frame):
        
        for people in self.peoplelist:
            
            #self.drawid(frame, people.person_id, people.pose2d.REye_15, people.pose2d.LEye_16, people.color)
            
            #self.drawpos(frame, people.pose2d.Nose_0, people.color)
            self.drawpos(frame, people.pose2d.Neck_1, people.color)
            #self.drawpos(frame, people.pose2d.RShoulder_2, people.color)
            #self.drawpos(frame, people.pose2d.RElbow_3, people.color)
            #self.drawpos(frame, people.pose2d.RWrist_4, people.color)
            #self.drawpos(frame, people.pose2d.LShoulder_5, people.color)
            #self.drawpos(frame, people.pose2d.LElbow_6, people.color)
            #self.drawpos(frame, people.pose2d.LWrist_7, people.color)
            
            #self.drawpos(frame, people.pose2d.RHip_9, people.color)
            #self.drawpos(frame, people.pose2d.RKnee_10, people.color)
            #self.drawpos(frame, people.pose2d.RAnkle_11, people.color)
            #self.drawpos(frame, people.pose2d.LHip_12, people.color)
            #self.drawpos(frame, people.pose2d.LKnee_13, people.color)
            #self.drawpos(frame, people.pose2d.LAnkle_14, people.color)
            #self.drawpos(frame, people.pose2d.REye_15, people.color)
            #self.drawpos(frame, people.pose2d.LEye_16, people.color)
            #self.drawpos(frame, people.pose2d.REar_17, people.color)
            #self.drawpos(frame, people.pose2d.LEar_18, people.color)
            #self.drawpos(frame, people.pose2d.LBigToe_19, people.color)
            #self.drawpos(frame, people.pose2d.LSmallToe_20, people.color)
            #self.drawpos(frame, people.pose2d.LHeel_21, people.color)
            #self.drawpos(frame, people.pose2d.RBigToe_22, people.color)
            #self.drawpos(frame, people.pose2d.RSmallToe_23, people.color)
            #self.drawpos(frame, people.pose2d.RHeel_24, people.color)

            #self.drawline(frame, people.pose2d.Nose_0, people.pose2d.Neck_1, people.color)
            #self.drawline(frame, people.pose2d.Nose_0, people.pose2d.LEye_16, people.color)
            #self.drawline(frame, people.pose2d.Nose_0, people.pose2d.REye_15, people.color)
            
            #self.drawline(frame, people.pose2d.LEye_16, people.pose2d.LEar_18, people.color)
            #self.drawline(frame, people.pose2d.REye_15, people.pose2d.REar_17, people.color)
            
            #self.drawline(frame, people.pose2d.Neck_1, people.pose2d.MidHip_8, people.color)
            
            
            #self.drawline(frame, people.pose2d.Neck_1, people.pose2d.RShoulder_2, people.color)
            #self.drawline(frame, people.pose2d.Neck_1, people.pose2d.LShoulder_5, people.color)
            
            #self.drawline(frame, people.pose2d.LShoulder_5, people.pose2d.LElbow_6, people.color)
            #self.drawline(frame, people.pose2d.LElbow_6, people.pose2d.LWrist_7, people.color)
            
            #self.drawline(frame, people.pose2d.RShoulder_2, people.pose2d.RElbow_3, people.color)
            #self.drawline(frame, people.pose2d.RElbow_3, people.pose2d.RWrist_4, people.color)
            
            #self.drawline(frame, people.pose2d.MidHip_8, people.pose2d.LHip_12, people.color)
            #self.drawline(frame, people.pose2d.LHip_12, people.pose2d.LKnee_13, people.color)
            #self.drawline(frame, people.pose2d.LKnee_13, people.pose2d.LAnkle_14, people.color)
            #self.drawline(frame, people.pose2d.LAnkle_14, people.pose2d.LBigToe_19, people.color)  
            #self.drawline(frame, people.pose2d.LAnkle_14, people.pose2d.LHeel_21, people.color)
            #self.drawline(frame, people.pose2d.LBigToe_19, people.pose2d.LSmallToe_20, people.color)
            
            #self.drawline(frame, people.pose2d.MidHip_8, people.pose2d.RHip_9, people.color)
            #self.drawline(frame, people.pose2d.RHip_9, people.pose2d.RKnee_10, people.color)
            #self.drawline(frame, people.pose2d.RKnee_10, people.pose2d.RAnkle_11, people.color)
            #self.drawline(frame, people.pose2d.RAnkle_11, people.pose2d.RHeel_24, people.color)
            #self.drawline(frame, people.pose2d.RAnkle_11, people.pose2d.RBigToe_22, people.color)
            #self.drawline(frame, people.pose2d.RBigToe_22, people.pose2d.RSmallToe_23, people.color)            
        return frame
    
    #def drawid(self, frame, text, pos1, pos2, color):
        #((h1, w1), c1) = self.getcenter(pos1)
        #((h2, w2), c2) = self.getcenter(pos2)
        #center = (int((h1 + h2) / 2), int((w1 + w2) / 2))
        #t = self.getthickness(c1, c2)
        #if (t > 0):
           # cv2.putText(frame, '{}'.format(text), center, int(cv2.FONT_HERSHEY_PLAIN), t, color, t, int(cv2.LINE_AA))
    
   
        
                
    def drawpos(self, frame, pos, color):
        ((h, w), c) = self.getcenter(pos)
        r = self.getradius(c)
        if (r > 0):
            cv2.circle(frame, (h, w), 0, color, -1)    
            
    def drawline(self, frame, pos1, pos2, color):
        ((h1, w1), c1) = self.getcenter(pos1)
        ((h2, w2), c2) = self.getcenter(pos2)
        t = self.getthickness(c1, c2)
        if (t > 0):
            cv2.line(frame, (h1, w1), (h2, w2), (255, 255, 255), t)
            
    def angle(self, frame, v1, v2):
        ((h1, w1), c1) = self.getcenter(v1)
        ((h2, w2), c2) = self.getcenter(v2)
        PT = [h1, w1, h2, w2]
        HH = [1,0,0,0]
        dx1 = PT[2] - PT[0]
        dy1 = PT[3] - PT[1]
        dx2 = HH[2] - HH[0]
        dy2 = HH[3] - HH[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = int(angle1 * 180/math.pi)
        # print(angle1)
        angle2 = math.atan2(dy2, dx2)
        angle2 = int(angle2 * 180/math.pi)
        # print(angle2)
        if angle1*angle2 >= 0:
            included_angle = abs(angle1-angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle            
        return included_angle
    
    def angleW(self, frame, v3, v4):
        ((x1, y1), c1) = self.getcenter(v3)
        ((x2, y2), c2) = self.getcenter(v4)
        LW = [x1, y1, x2, y2]
        dx1 = abs(LW[0] - LW[2])
        dy1 = abs(LW[1] - LW[3])
        if dx1 >= dy1:
           LWtest = 0
        else:
           LWtest = 1
        return LWtest
    
    def getcenter(self, pos):
        ((h, w), c)  = pos
        return ((int(h), int(w)), c)
    
    def getradius(self, c):
        r = self.circleradius if (c > self.confidencethreshold) else 0
        return r
    
    def getthickness(self, c1, c2):
        t = self.linewidth if ((c1 > self.confidencethreshold) and (c2 > self.confidencethreshold)) else 0
        return t    
     
root = 'D:\LAB5-openpose\5videosJSON'

#video = '1_02_02.mp4'
#video = '1_02_05.mp4'
#video = '1_06_07.mp4'
#video = '1_09_09.mp4'
#video = '3_07_05.mp4'
video = 'D:\DL_final_test\falldown20210102B.mp4'


jsonpath1 = 'D:/LAB5-openpose/5videosJSON/JSON_falldown20210102B'
videopath = 'D:/DL_final_test/falldown20210102B.mp4'
(videoname, videoext) = os.path.splitext(video)
outputpath = 'D:/DL_final_test/falldown20210102B.out.avi'

#path = '{}/1_02_02/1_02_02_000000000000_keypoints.json'
jsonpath2 = 'D:/LAB5-openpose/5videosJSON/JSON_falldown20210102B/*.json'.format(jsonpath1, videoname)

flist = glob.glob(jsonpath2)
flist.sort()

cap = cv2.VideoCapture(videopath)
fps = cap.get(cv2.CAP_PROP_FPS)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outvideo = cv2.VideoWriter(outputpath, fourcc, fps, (w, h), True)

kp = None
fcnt = -1
counterT = 0
counterN = 0

# Read video frame by frame
while True:
    # Get 1 frame
    success, frame = cap.read()

    if success:
        
        fcnt += 1
        fname = flist[fcnt]
        f = open(fname, 'r')
        jsonstr = f.read()
        f.close()
        peopledict = jsonpickle.decode(jsonstr)
        fkp = KeyPoints(peopledict)
        
        if not (kp is None):
            kp = kp.traceid(fkp)
        else:
            kp = fkp.initid()

        skframe = kp.drawsk(frame)
        cv2.rectangle(skframe, (930, 0), (1300, 750), (0, 255, 0), -1)
        angleT = kp.angle(skframe, Neck_1, MidHip_8)   
        print('angleT=', angleT)
        
        LWtest1 = kp.angleW(skframe, Neck_1, MidHip_8)
        print('LWtest1=', LWtest1)
        
        if angleT <= 40:
            counterT = counterT +1
            if counterT >= 40:
                counterN = 0
                cv2.putText(skframe, 'The person is dangerous', (20, 60), int(cv2.FONT_HERSHEY_PLAIN), 3.5, (0, 0, 255), 3, int(cv2.LINE_8))
        else:
            counterN = counterN +1
            if counterN >= 60:
                counterT = 0
        
        print('counterT=', counterT)
        print('counterN=', counterN)

                # Write 1 frame to output video
        outvideo.write(skframe)
    else:
        break

# Release resource
cap.release()
outvideo.release()
