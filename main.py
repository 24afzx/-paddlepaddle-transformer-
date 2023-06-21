import tkinter as tk
from predict import do_predict
import threading

def translate_text():
    # 禁止"翻译"按钮，避免用户重复点击
    button['state'] = 'disabled'
    # 启动翻译操作
    # 启动翻译操作
    threading.Thread(target=run_translation).start()

def run_translation():
    input_text = src_textbox.get()
    output_text = do_predict([input_text])[0]
    print('-' * 100)
    print(f'原文: {input_text}')
    print(f'译文: {output_text}')
    tgt_textbox.config(state='normal')
    # 清空输出文本框
    tgt_textbox.delete(1.0, 'end')
    # 在GUI线程中更新显示结果
    tgt_textbox.insert(1.0, output_text)
    # 在翻译完毕后恢复“翻译”按钮
    button.config(state='normal')

def create_label(text, row, column, padx=5, pady=5, sticky="w"):
    label = tk.Label(root, text=text)
    label.grid(row=row, column=column, padx=padx, pady=pady, sticky=sticky)
    return label


def create_text_box(width, height, row, column, padx=5, pady=5, **kwargs):
    text_box = tk.Text(root, width=width, height=height, **kwargs)
    text_box.grid(row=row, column=column, padx=padx, pady=pady)
    return text_box


def create_entry_box(width, row, column, padx=5, pady=5):
    entry_box = tk.Entry(root, width=width)
    entry_box.grid(row=row, column=column, padx=padx, pady=pady)
    return entry_box


root = tk.Tk()
root.title('中德机器翻译')
root.geometry("500x250")

create_label("输入内容：", 0, 0)
src_textbox = create_entry_box(50, 0, 1)

create_label("翻译结果：", 1, 0)
var_output = tk.StringVar()
tgt_textbox = create_text_box(50, 5, 1, 1, state='disabled')


button = tk.Button(root, text="翻译", command=translate_text)
button.grid(row=2, column=1, padx=5, pady=5)

root.mainloop()