# Copy3_3.24
critical 学习率 0.001

tau 0.0001

现在save的时候reward的图也会一起被save下来

        r += 10
        r = r * 4

# Copy3
网络改成了4个1024层并且修复了之前的一些构建网络地方的代码错误

经验池大小改成了20000

每个epi的maxstep设为了35

模型在carbinet_1000epi_four1024layers

