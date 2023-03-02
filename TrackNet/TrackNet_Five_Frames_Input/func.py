import cv2
import numpy as np
import pyshine as ps

def info(output_img, need_L_B, need_R_B, bounce_in_round, event_q, attack):
    output_img = cv2.putText(output_img, 'state:', (760, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (138, 241, 255), 2, cv2.LINE_AA)
    if need_L_B:
        output_img = cv2.putText(output_img, 'need left bounce', (880, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (138, 241, 255), 2, cv2.LINE_AA)
    elif need_R_B:
        output_img = cv2.putText(output_img, 'need right bounce', (880, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (138, 241, 255), 2, cv2.LINE_AA)
    else:
        output_img = cv2.putText(output_img, 'wait', (880, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (138, 241, 255), 2, cv2.LINE_AA)
    
    output_img = cv2.putText(output_img, 'previous event:', (760, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (138, 241, 255), 2, cv2.LINE_AA)
    output_img = cv2.putText(output_img, event_q[0], (1050, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (138, 241, 255), 2, cv2.LINE_AA)
    
    output_img = cv2.putText(output_img, 'bounce in round:', (760, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (138, 241, 255), 2, cv2.LINE_AA)
    output_img = cv2.putText(output_img, str(bounce_in_round), (1080, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (138, 241, 255), 2, cv2.LINE_AA)
    
    if attack[1]:
        output_img = cv2.putText(output_img, 'right attack', (1660, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        output_img = cv2.putText(output_img, 'right attack', (1660, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if attack[0]:
        output_img = cv2.putText(output_img, 'left attack', (1660, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        output_img = cv2.putText(output_img, 'left attack', (1660, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return output_img

def draw_out(q, need_L_B, need_R_B, point, output_img, event, bounce_in_round, event_q, attack):

    b, n = 0.999, 0.99
    
    for i in range(0,17):
        if q[i] is not None:
            draw_x = q[i][0]
            draw_y = q[i][1]
            output_img = cv2.circle(output_img, tuple([draw_x, draw_y]), 5, (255, 0, 255), -1)
    
    output_img = ps.putBText(output_img, '                ',755, 60, 50, 15, alpha=0.5, font_scale=2, background_RGB=(0,0,0), text_RGB=(255,0,0))
    output_img = info(output_img, need_L_B, need_R_B, bounce_in_round, event_q, attack)
    
    
    if event[0] >= n:
        output_img = cv2.putText(output_img, 'over net', (1500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        output_img = cv2.putText(output_img, 'bounce', (1500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        output_img = cv2.putText(output_img, 'empty', (1500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    elif event[1] >= b:
        output_img = cv2.putText(output_img, 'over net', (1500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        output_img = cv2.putText(output_img, 'bounce', (1500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        output_img = cv2.putText(output_img, 'empty', (1500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        output_img = cv2.putText(output_img, 'over net', (1500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        output_img = cv2.putText(output_img, 'bounce', (1500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        output_img = cv2.putText(output_img, 'empty', (1500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    #output_img = cv2.putText(output_img, str(event), (900, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    output_img = ps.putBText(output_img, '              ',30, 90, 50, 15, alpha=0.5, font_scale=2, background_RGB=(0,0,0), text_RGB=(255,0,0))
    output_img = cv2.putText(output_img, str(point[0][1]) + ":" + str(point[1][1]), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 3, cv2.LINE_AA)
    output_img = cv2.putText(output_img, str(point[0][0]) + "       " + str(point[1][0]), (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (245, 75, 240), 3, cv2.LINE_AA)
    
    return output_img
    
def event_caculation(x, y, max_y, event, need_L_B, need_R_B, bounce_count, serve, point, bounce_q, bounce_in_round, event_q, attack):
    b, n = 0.999, 0.99
    if event[1] >= b:
        bounce_q += 1
    else:
        bounce_q = 0
        
    if event[0] >= n:
        event_q.appendleft("over net")
    
    if bounce_q > 1:
        bounce_count = 0
    
    if bounce_q == 1 and serve and bounce_count > 12:
        if need_L_B and x <= 960:
            bounce_count = 0
            bounce_in_round += 1
            need_L_B = False
            need_R_B = True
            event_q.appendleft("bounce left")
        elif need_R_B and x > 960:
            bounce_in_round += 1
            bounce_count = 0
            need_R_B = False
            need_L_B = True
            event_q.appendleft("bounce right")
        else:
            bounce_count = 0
            serve = False
            if need_L_B:
                point[0][1] += 1
                print("Left player win!")
                attack[0] = False
                attack[1] = False
            elif need_R_B:
                point[1][1] += 1
                print("Right player win!")
                attack[0] = False
                attack[1] = False
            need_L_B = False
            need_R_B = False
            bounce_in_round = 0
            
    if event[1] >= b and (y - max_y) > 240 and not serve and y <= 880:
        print("Serve start at", x,y)
        if x > 960:
            need_L_B = True
            event_q.appendleft("bounce right")
            attack[1] = True
        else:
            need_R_B = True
            event_q.appendleft("bounce left")
            attack[0] = True
        serve = True
        bounce_count = 0
        bounce_in_round += 1
        max_y = 1080

    if serve and  bounce_count > 80:
        bounce_count = 0
        bounce_in_round = 0
        serve = False
        if need_L_B:
            point[0][1] += 1
            print("Left player win!")
            attack[0] = False
            attack[1] = False
        elif need_R_B:
            point[1][1] += 1
            print("Right player win!")
            attack[0] = False
            attack[1] = False
        need_L_B = False
        need_R_B = False
    
    bounce_count += 1
    
    return max_y, need_L_B, need_R_B, bounce_count, serve, point, bounce_in_round, bounce_q
    
def predict_ball(pr, width, height, output_width, output_height, n_classes):
    pr = pr.reshape((height,  width, n_classes)).argmax(axis=2)
    pr = pr.astype(np.uint8) 
    heatmap = cv2.resize(pr, (output_width, output_height))
    ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=4,maxRadius=8)
    
    return circles
    
def get_ball(circles, q):
    x, y = None, None
    if circles is not None:
        #if only one tennis be detected
        if len(circles) == 1:
            x = int(circles[0][0][0])
            y = int(circles[0][0][1])
            
            q.appendleft([x,y])
            q.pop()
        else:
            q.appendleft(None)
            q.pop()
    else:
        q.appendleft(None)
        q.pop()
    return x, y
    
