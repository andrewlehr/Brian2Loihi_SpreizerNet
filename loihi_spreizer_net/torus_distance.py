from brian2 import *
@implementation('cython', '''
    cdef double torus_distance(double x_pre, double x_post, double y_pre, double y_post):
        x_pre = x_pre % 1
        y_pre = y_pre % 1

        cdef double dx = abs(x_pre - x_post)
        cdef double dy = abs(y_pre - y_post)
        
        if dx > 0.5:
            dx = 1 - dx
            
        if dy > 0.5:
            dy = 1 - dy
            
        return sqrt(dx*dx + dy*dy)
    ''')
@check_units(x_pre=1, x_post=1, y_pre=1, y_post=1, result=1)
def torus_distance(x_pre, x_post, y_pre, y_post):
    x_pre = x_pre % 1
    y_pre = y_pre % 1

    dx = abs(x_pre - x_post)
    dy = abs(y_pre - y_post)

    if dx > 0.5:
        dx = 1 - dx

    if dy > 0.5:
        dy = 1 - dy

    return sqrt(dx * dx + dy * dy)