import queue
import func
q = queue.deque(maxlen=5)
for i in range(0,5):
    q.appendleft(i)
    
print(q)
func.aa(q)
print(q)