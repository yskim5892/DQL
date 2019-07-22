from tkinter import *

def coordinate_transform(w, h, map_x, size):
    margin = 1/6
    return (w * margin + map_x[0] * ((1- 2 * margin) * w  /  size), h * (1 -  margin) - map_x[1] * ((1- 2 * margin) * h / size))


def rgb_to_color(rgb):
    return "#%02x%02x%02x" % rgb

def draw_block(canvas, block, coord, pixel_w, pixel_h):
    mw = pixel_w / 10
    mh = pixel_h / 10
    outline_width = pixel_w / 10
    if (block == 27):
        return
    if (block > 27):
        color = rgb_to_color(tuple([int(x * (((block - 28)%3) + 2) / 4) for x in (255, 255, 255)]))

        return canvas.create_polygon(coord[0] + mw, coord[1] + mh, coord[0] + pixel_w - mw, coord[1] + mh, coord[0] + pixel_w - mw,
                              coord[1] + pixel_h - mh, coord[0] + mw, coord[1] + pixel_h - mh, fill=color,
                              outline='white', width = outline_width)

    color = (0, 0, 0)
    if (block // 9 == 0):
        color = (0, 0, 255)
    elif (block // 9 == 1):
        color = (255, 0, 0)
    elif (block // 9 == 2):
        color = (255, 255, 0)
    fill_color = rgb_to_color(tuple([int(x * (block % 3) / 2) for x in color]))
    color = rgb_to_color(color)

    ow = outline_width
    if block % 3 == 1:
        ow = 0


    if ((block % 9) // 3 == 0):
        return canvas.create_oval(coord[0] + mw, coord[1] + pixel_h - mh, coord[0] + pixel_w - mw, coord[1] + mh, fill=fill_color, \
                           outline=color, width=ow)
    elif ((block % 9) // 3 == 1):
        return canvas.create_polygon(coord[0] + mw, coord[1] + mh, coord[0] + pixel_w - mw, coord[1] + mh, coord[0] + pixel_w - mw,
                              coord[1] + pixel_h - mh, coord[0] + mw, coord[1] + pixel_h - mh, fill=fill_color,
                              outline=color, width=ow)
    elif ((block % 9) // 3 == 2):
        return canvas.create_polygon(coord[0] + mw, coord[1] + pixel_h - mh, coord[0] + pixel_w - mw, coord[1] + pixel_h - mh, \
                              coord[0] + pixel_w / 2, coord[1] + mh, fill=fill_color, outline=color, width=ow)

def BHB_display(step, size, trajectory):
    window = Tk()
    w = 900
    h = 900
    window.title("Bauhausbreak_step" + str(step))
    window.geometry("900x900+100+100")
    window.resizable(False, False)
    canvas = Canvas(window, relief='solid', bd=2)

    pixel_w = coordinate_transform(w, h, (1, 0), size)[0] - coordinate_transform(w, h, (0, 0), size)[0]
    pixel_h = coordinate_transform(w, h, (0, 1), size)[1] - coordinate_transform(w, h, (0, 0), size)[1]

    pixels = dict()
    def draw(state):
        canvas.delete('all')
        for x in range(0, size):
            for y in range(0, size):
                block = state.blocks[x][y]
                coord = coordinate_transform(w, h, (x, y), size)

                canvas.create_polygon(coord[0], coord[1], coord[0] + pixel_w, coord[1], coord[0] + pixel_w,
                                      coord[1] + pixel_h, coord[0], coord[1] + pixel_h, fill='black',
                                      outline='white')
                if(block != 27):
                    draw_block(canvas, block, coord, pixel_w, pixel_h)


        for x in range(0, 8):
            coord = coordinate_transform(w, h, (x, -1), size)
            if(x > 7 - state.gauge):
                canvas.create_oval(coord[0] + pixel_w * 7 / 16, coord[1] + pixel_h * 7 / 16, coord[0] + pixel_w * 9 / 16, coord[1] + pixel_h * 9 / 16,\
                                 fill='black', outline='black')
            else:
                canvas.create_polygon(coord[0] + pixel_w * 1 / 3, coord[1] + pixel_h * 1 / 3, coord[0] + pixel_w * 1 / 3, coord[1] + pixel_h * 2 / 3,\
                                  coord[0] + pixel_w * 2 / 3, coord[1] + pixel_h * 2 / 3, coord[0] + pixel_w * 2 / 3, coord[1] + pixel_h * 1 / 3,\
                                  fill='black', outline='black')

    current_block = None
    def draw_current_block(block, pos, a):
        global current_block
        if a == 0:
            coord = coordinate_transform(w, h, (0, size), size)
            current_block = draw_block(canvas, block, coord, pixel_w, pixel_h)
        else:
            coord = coordinate_transform(w, h, (pos, size), size)
            canvas.delete(current_block)
            current_block = draw_block(canvas, block, coord, pixel_w, pixel_h)

    i = 0
    canvas.pack(fill=BOTH, expand=1)
    print([x[1] for x in trajectory])
    T = 1000
    for i in range(len(trajectory)):
        state = trajectory[i][0]
        action = trajectory[i][1]
        window.after(2 * T * (i + 1), draw, state)
        window.after(2 * T * (i + 1), draw_current_block, state.current_block, action, 0)
        window.after(2 * T * (i + 1) + T, draw_current_block, state.current_block, action, 1)
    window.mainloop()