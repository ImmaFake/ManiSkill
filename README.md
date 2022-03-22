# PLOT方法更改
    def plot_rewards(self, scatter):
        y = self.Reward_record
        x = range(len(y))
        plt.figure(figsize=(10, 10), dpi=70)
        
        if scatter:
            plt.scatter(x, y)
        else:
            plt.plot(x, y)
        plt.show()
# 新版Copy2（我一不小心给原版的覆盖了-，-你们再建个文件夹或者改改名字啥的把上一版传上去把
能把pretrain model load进去了，然后上面的plot也改过来了。
以上来没训练的时候特别正常，问题就是一但开始训练他就开始转圈圈。
