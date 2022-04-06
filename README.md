#画图

Reward:

    data = pd.read_csv('./DPG model/Check_Points/8000epi/Rewards.csv')
    y = np.array(data).squeeze()
    x = range(y.shape[0])
    size = 50
    z = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] >= 300:
            y[i] = 0  
    for i in range(len(y)):
        if i > (size/2 -1) and i < (len(y) - size/2):  
            for j in range(size):
                z[i] += y[i + j - int(size/2)]  #平均reward        
        z[i] = z[i]/size
    plt.figure(figsize=(10, 10), dpi=70)
    plt.plot(x, y)
    plt.plot(x, z)
    plt.legend(['y','z'])
    plt.show()
    print('maximum mean rewards: ', np.argmax(z))

Success:

data = pd.read_csv('./DPG model/cat_notfix/Success_record.csv')
y = np.array(data).squeeze()
x = range(y.shape[0])
m = np.zeros(y.shape[0])

for i in range(50,7950):
    for j in range(50):
        if y[i-50+j] == 1:
            m[i] += 1
    m[i] = m[i] / 50   
        
plt.figure(figsize=(10, 10), dpi=70)
plt.plot(x, m)
plt.show()
print(max(m))#最大值
print(np.argmax(m))#最大值所在位置

# whp_3.25
传上去了两个无噪音无learn纯测试的代码，在test文件夹里面

更新了cat的代码，step改成75后结果更好了

新加入了check points， 每50epi存一次，需要在DPG model下新建一个Check_Points的文件夹。 存图太多回报错，不过不影响

新加入了一个array用来存是否success（1/0），对应的成功率画图代码还没写，目前的这个太乱了挤在一起看不出来，只能看线的密度
# whp_3.25

加入OU噪音以替代高斯

reward中加入抽屉开合程度

        r += info.get('qpos')*5000

尝试过*100，*1000，*5000。仅于*5000时可以自行success

在render时发现完全打开但是不算做success的情况，查看done和success状态均为False

这个打开柜子的方式不是拉把手而是从左侧扒开。大部分时候只能扒开一点，有时会扒开下面的柜门。可能原因为给柜子开合状态的reward高了一点，将机械臂与把手距离的reward盖住了

发现如果部分动作reward设的超级高（如success后reward过高），可能是导致转圈的原因

notebook：cat

folder：carbinet_OU3

当前参数：   

        step = 50
        sigma decay = 1/100000 (per step) #var = 5
        lra = 0.001
        lrc = 0.001
        gamma = 0.9
        tau = 0.01
        memory_capacity = 10000
        num_layer  = 4
        hiddden_nodes = 256
        pretrain = False
        
        

# fyf_3.25_记录3

reward 421 -> 368

失败

当前参数：

        step = 100
        var = 5
        lra = 0.001
        lrc = 0.001
        gamma = 0.9
        tau = 0.01
        memory_capacity = 10000
        num_layer  = 3
        hiddden_nodes = 512
        pretrain = False
 

目前最佳 step = 75
# fyf_3.25_记录2

reward 388 -> 421

当前参数：

        step = 75
        var = 5
        lra = 0.001
        lrc = 0.001
        gamma = 0.9
        tau = 0.01
        memory_capacity = 10000
        num_layer  = 3
        hiddden_nodes = 512
        pretrain = False

# fyf_3.25_记录1

reward: 380 -> 388

当前参数：

        step = 40
        var = 5
        lra = 0.001
        lrc = 0.001
        gamma = 0.9
        tau = 0.01
        memory_capacity = 10000
        num_layer  = 3
        hiddden_nodes = 512
        pretrain = False


# dpg_fyf_3.24_script2(附结果)
可以把hidden nodes 当成参数

自训练可用，pretrain不可用，好像确实在learn，还没训完，训完传model

已上传3000轮的模型cabinet_selftrain_3000eps

已上传30000轮的模型cabinet_selftrain_30000eps,失败了，maximum mean reward = 380

当前参数：
        
        step = 35
        var = 5
        lra = 0.001
        lrc = 0.001
        gamma = 0.9
        tau = 0.01
        memory_capacity = 10000
        num_layer  = 3
        hiddden_nodes = 256
        pretrain = False

# 找到一个csdn
https://blog.csdn.net/qq_37395293/article/details/114226081?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164811933916782248523306%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164811933916782248523306&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-3-114226081.142^v3^pc_search_result_control_group,143^v4^control&utm_term=ddpg&spm=1018.2226.3001.4187

里面的噪声添加好像比单纯的var更高级一点


# 可能是可以调用显卡的方式

        use_cuda = torch.cuda.is_available()
        device   = torch.device("cuda" if use_cuda else "cpu")


# dpg_3.24
critic network 进行pretrain
hidden nodes: 256 512 256

# Copy3_3.24
critical 学习率 0.001

tau 0.0001

现在save的时候reward的图也会一起被save下来

        r += 10
        r = r * 4


-10.091069829064672
-8.482848429646328
-7.17262549225644
-6.251941905221525
-5.637000860832369
-5.17555508979571
-4.795561710131981
-4.613756278298448
-4.418248861922317
-4.131623378200452
-3.8814617579030664
-3.6391049670507414
-3.708595358379071
-3.8314774116644834
-3.8722980859398106
-4.041868872535481
-4.178343141394123
-5.151454245355239
-6.206955193408271
-7.071980125167888
-7.945841513148057
-8.688644349065274
-9.631269272766332
-10.230128660194653
-11.56082410260444
-11.899343430258039
-12.650781807507489
-13.12216629445909
-14.136213924247917
-14.715582603949741
-15.626390350555791
-16.29811065104297
SUCCESS
69983.0169031656
Episode: 14  Reward: 69730 Explore: 0.10

-10.227783285130956
-8.737858004389977
-7.490684074512693
-6.681493523549722
-6.106999946165388
-5.460176653790034
-5.000583478749826
-4.650343061948886
-4.539363737579658
-4.342334429483721
-3.6433019136794265
-3.015501781523543
-3.2501725746219936
-4.3883222866106415
-6.519440411079852
-8.569781200966869
-10.338091187619339
-11.403949718444935
-12.464690322146236
-13.201661311140203
-14.102103806731767
-14.943248481257744
-15.169497634040095
-15.86011071399507
-16.39358843665733
-16.630835725978116
-16.40435030437186
-16.66464902025776
-16.747125835369026
-17.45291924638056
-17.53975549894903
-18.12702167210071
-18.446940965430557
-19.22269335120591
-19.43900808662312
-20.008872127216375
Episode: 13  Reward: -413 Explore: 0.10

# Copy3
网络改成了4个1024层并且修复了之前的一些构建网络地方的代码错误

经验池大小改成了20000

每个epi的maxstep设为了35

模型在carbinet_1000epi_four1024layers

