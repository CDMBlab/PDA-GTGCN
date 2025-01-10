import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

class MyGUI:
    def __init__(self, master):
        self.master = master
        master.title("PDA-GTGCN计算相似性")

        self.input_label = tk.Label(master, text="选择本地文件:")
        self.input_label.grid(row=0, column=0, pady=5)

        self.input_entry = tk.Entry(master)
        self.input_entry.grid(row=0, column=1, pady=5)

        self.button_browse = tk.Button(master, text="浏览", command=self.browse_file)
        self.button_browse.grid(row=0, column=2, pady=5)

        self.button1 = tk.Button(master, text="piRNA序列相似性", command=self.calculate_circRNA_func_similarity)
        self.button1.grid(row=1, column=0, pady=5)

        self.button2 = tk.Button(master, text="piRNA高斯核相似性", command=self.calculate_circRNA_gaussian_similarity)
        self.button2.grid(row=1, column=1, pady=5)

        self.button3 = tk.Button(master, text="疾病语义相似性", command=self.calculate_disease_semantic_similarity)
        self.button3.grid(row=2, column=0, pady=5)

        self.button4 = tk.Button(master, text="疾病高斯核相似性", command=self.calculate_disease_gaussian_similarity)
        self.button4.grid(row=2, column=1, pady=5)

        self.switch_button = tk.Button(master, text="切换窗口", command=self.switch_window)
        self.switch_button.grid(row=6, column=1, pady=5, sticky=tk.SE)  # 右下角对齐

        self.output_label = tk.Label(master, text="输出结果:")
        self.output_label.grid(row=4, column=0, columnspan=2, pady=5)

        self.output_text = tk.Text(master, height=10, width=50)
        self.output_text.grid(row=5, column=0, columnspan=2, pady=5)

    def display_output(self, result):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, result)

    def browse_file(self):
        file_path = filedialog.askopenfilename(title="选择本地文件", filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])

        if file_path:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, file_path)
            result = f"已选择文件：{file_path}"
            self.display_output(result)

    def calculate_circRNA_func_similarity(self):
        result = "生成piRNA序列相似性矩阵"
        self.display_output(result)

    def calculate_circRNA_gaussian_similarity(self):
        result = "生成piRNA高斯核相似性矩阵"
        self.display_output(result)

    def calculate_disease_semantic_similarity(self):
        result = "生成疾病语义相似性矩阵"
        self.display_output(result)

    def calculate_disease_gaussian_similarity(self):
        result = "生成疾病高斯核相似性矩阵"
        self.display_output(result)

    def switch_window(self):
        new_window = tk.Toplevel(self.master)
        new_gui = SecondaryGUI(new_window, self.input_entry.get())

class SecondaryGUI:
    def __init__(self, master, file_path):
        self.master = master
        master.title("PDA-GTGCN特征提取和关联预测")

        self.file_path = file_path

        self.button_multiview = tk.Button(master, text="piRNA特征表示", command=self.calculate_multiview)
        self.button_multiview.grid(row=0, column=0, pady=5)

        self.button_embed_circRNA = tk.Button(master, text="疾病特征表示", command=self.calculate_embed_circRNA)
        self.button_embed_circRNA.grid(row=0, column=1, pady=5)

        self.button_embed_disease = tk.Button(master, text="异构网络", command=self.calculate_embed_disease)
        self.button_embed_disease.grid(row=1, column=0, pady=5)

        self.button_feature_similarity = tk.Button(master, text="图卷积网络", command=self.calculate_feature_similarity)
        self.button_feature_similarity.grid(row=1, column=1, pady=5)

        self.button_random_perturbation = tk.Button(master, text="特征分组", command=self.build_random_perturbation)
        self.button_random_perturbation.grid(row=2, column=0, pady=5)

        self.button_network_fusion = tk.Button(master, text="特征转换网络", command=self.network_fusion)
        self.button_network_fusion.grid(row=2, column=1, pady=5)

        self.button_high_order_feature = tk.Button(master, text="高阶特征提取", command=self.high_order_feature_extraction)
        self.button_high_order_feature.grid(row=3, column=0, pady=5)

        self.button_association_prediction = tk.Button(master, text="关联预测", command=self.association_prediction)
        self.button_association_prediction.grid(row=3, column=1, pady=5)

        self.output_label = tk.Label(master, text="输出结果:")
        self.output_label.grid(row=4, column=0, columnspan=2, pady=5)

        self.output_text = tk.Text(master, height=10, width=50)
        self.output_text.grid(row=5, column=0, columnspan=2, pady=5)

    def display_output(self, result):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, result)

    def calculate_multiview(self):
        result = "计算piRNA特征表示"
        self.display_output(result)

    def calculate_embed_circRNA(self):
        result = "计算疾病特征表示"
        self.display_output(result)

    def calculate_embed_disease(self):
        result = "构建异构网络"
        self.display_output(result)

    def calculate_feature_similarity(self):
        result = "执行图卷积操作"
        self.display_output(result)

    def build_random_perturbation(self):
        result = "执行特征分组操作"
        self.display_output(result)

    def network_fusion(self):
        result = "执行特征转换操作"
        self.display_output(result)

    def high_order_feature_extraction(self):
        result = "提取piRNA和疾病的高阶特征"
        self.display_output(result)

    def association_prediction(self):
        result = "生成关联得分矩阵，完成预测"
        self.display_output(result)


# 创建主窗口
root = tk.Tk()
root.geometry("400x350")  # 设置窗口大小
my_gui = MyGUI(root)
root.mainloop()
