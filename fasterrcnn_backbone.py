name                                    shape               #elements      (M)    (MB)  trainable    dtype
--------------------------------------  ------------------  -----------  -----  ------  -----------  -------------
backbone/conv1/weight                   [64, 3, 3, 3]       1,728         0       0.01  False        torch.float32
backbone/bn1/weight                     [64]                64            0       0     False        torch.float32
backbone/bn1/bias                       [64]                64            0       0     False        torch.float32
backbone/layer1/0/conv1/weight          [64, 64, 1, 1]      4,096         0       0.02  False        torch.float32
backbone/layer1/0/bn1/weight            [64]                64            0       0     False        torch.float32
backbone/layer1/0/bn1/bias              [64]                64            0       0     False        torch.float32
backbone/layer1/0/conv2/weight          [64, 64, 3, 3]      36,864        0.04    0.14  False        torch.float32
backbone/layer1/0/bn2/weight            [64]                64            0       0     False        torch.float32
backbone/layer1/0/bn2/bias              [64]                64            0       0     False        torch.float32
backbone/layer1/0/conv3/weight          [256, 64, 1, 1]     16,384        0.02    0.06  False        torch.float32
backbone/layer1/0/bn3/weight            [256]               256           0       0     False        torch.float32
backbone/layer1/0/bn3/bias              [256]               256           0       0     False        torch.float32
backbone/layer1/0/downsample/0/weight   [256, 64, 1, 1]     16,384        0.02    0.06  False        torch.float32
backbone/layer1/0/downsample/1/weight   [256]               256           0       0     False        torch.float32
backbone/layer1/0/downsample/1/bias     [256]               256           0       0     False        torch.float32
backbone/layer1/1/conv1/weight          [64, 256, 1, 1]     16,384        0.02    0.06  False        torch.float32
backbone/layer1/1/bn1/weight            [64]                64            0       0     False        torch.float32
backbone/layer1/1/bn1/bias              [64]                64            0       0     False        torch.float32
backbone/layer1/1/conv2/weight          [64, 64, 3, 3]      36,864        0.04    0.14  False        torch.float32
backbone/layer1/1/bn2/weight            [64]                64            0       0     False        torch.float32
backbone/layer1/1/bn2/bias              [64]                64            0       0     False        torch.float32
backbone/layer1/1/conv3/weight          [256, 64, 1, 1]     16,384        0.02    0.06  False        torch.float32
backbone/layer1/1/bn3/weight            [256]               256           0       0     False        torch.float32
backbone/layer1/1/bn3/bias              [256]               256           0       0     False        torch.float32
backbone/layer1/2/conv1/weight          [64, 256, 1, 1]     16,384        0.02    0.06  False        torch.float32
backbone/layer1/2/bn1/weight            [64]                64            0       0     False        torch.float32
backbone/layer1/2/bn1/bias              [64]                64            0       0     False        torch.float32
backbone/layer1/2/conv2/weight          [64, 64, 3, 3]      36,864        0.04    0.14  False        torch.float32
backbone/layer1/2/bn2/weight            [64]                64            0       0     False        torch.float32
backbone/layer1/2/bn2/bias              [64]                64            0       0     False        torch.float32
backbone/layer1/2/conv3/weight          [256, 64, 1, 1]     16,384        0.02    0.06  False        torch.float32
backbone/layer1/2/bn3/weight            [256]               256           0       0     False        torch.float32
backbone/layer1/2/bn3/bias              [256]               256           0       0     False        torch.float32
backbone/layer2/0/conv1/weight          [128, 256, 1, 1]    32,768        0.03    0.12  True         torch.float32
backbone/layer2/0/bn1/weight            [128]               128           0       0     True         torch.float32
backbone/layer2/0/bn1/bias              [128]               128           0       0     True         torch.float32
backbone/layer2/0/conv2/weight          [128, 128, 3, 3]    147,456       0.14    0.56  True         torch.float32
backbone/layer2/0/bn2/weight            [128]               128           0       0     True         torch.float32
backbone/layer2/0/bn2/bias              [128]               128           0       0     True         torch.float32
backbone/layer2/0/conv3/weight          [512, 128, 1, 1]    65,536        0.06    0.25  True         torch.float32
backbone/layer2/0/bn3/weight            [512]               512           0       0     True         torch.float32
backbone/layer2/0/bn3/bias              [512]               512           0       0     True         torch.float32
backbone/layer2/0/downsample/0/weight   [512, 256, 1, 1]    131,072       0.12    0.5   True         torch.float32
backbone/layer2/0/downsample/1/weight   [512]               512           0       0     True         torch.float32
backbone/layer2/0/downsample/1/bias     [512]               512           0       0     True         torch.float32
backbone/layer2/1/conv1/weight          [128, 512, 1, 1]    65,536        0.06    0.25  True         torch.float32
backbone/layer2/1/bn1/weight            [128]               128           0       0     True         torch.float32
backbone/layer2/1/bn1/bias              [128]               128           0       0     True         torch.float32
backbone/layer2/1/conv2/weight          [128, 128, 3, 3]    147,456       0.14    0.56  True         torch.float32
backbone/layer2/1/bn2/weight            [128]               128           0       0     True         torch.float32
backbone/layer2/1/bn2/bias              [128]               128           0       0     True         torch.float32
backbone/layer2/1/conv3/weight          [512, 128, 1, 1]    65,536        0.06    0.25  True         torch.float32
backbone/layer2/1/bn3/weight            [512]               512           0       0     True         torch.float32
backbone/layer2/1/bn3/bias              [512]               512           0       0     True         torch.float32
backbone/layer2/2/conv1/weight          [128, 512, 1, 1]    65,536        0.06    0.25  True         torch.float32
backbone/layer2/2/bn1/weight            [128]               128           0       0     True         torch.float32
backbone/layer2/2/bn1/bias              [128]               128           0       0     True         torch.float32
backbone/layer2/2/conv2/weight          [128, 128, 3, 3]    147,456       0.14    0.56  True         torch.float32
backbone/layer2/2/bn2/weight            [128]               128           0       0     True         torch.float32
backbone/layer2/2/bn2/bias              [128]               128           0       0     True         torch.float32
backbone/layer2/2/conv3/weight          [512, 128, 1, 1]    65,536        0.06    0.25  True         torch.float32
backbone/layer2/2/bn3/weight            [512]               512           0       0     True         torch.float32
backbone/layer2/2/bn3/bias              [512]               512           0       0     True         torch.float32
backbone/layer2/3/conv1/weight          [128, 512, 1, 1]    65,536        0.06    0.25  True         torch.float32
backbone/layer2/3/bn1/weight            [128]               128           0       0     True         torch.float32
backbone/layer2/3/bn1/bias              [128]               128           0       0     True         torch.float32
backbone/layer2/3/conv2/weight          [128, 128, 3, 3]    147,456       0.14    0.56  True         torch.float32
backbone/layer2/3/bn2/weight            [128]               128           0       0     True         torch.float32
backbone/layer2/3/bn2/bias              [128]               128           0       0     True         torch.float32
backbone/layer2/3/conv3/weight          [512, 128, 1, 1]    65,536        0.06    0.25  True         torch.float32
backbone/layer2/3/bn3/weight            [512]               512           0       0     True         torch.float32
backbone/layer2/3/bn3/bias              [512]               512           0       0     True         torch.float32
backbone/layer3/0/conv1/weight          [256, 512, 1, 1]    131,072       0.12    0.5   True         torch.float32
backbone/layer3/0/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/0/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/0/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/0/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/0/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/0/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/0/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/0/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/0/downsample/0/weight   [1024, 512, 1, 1]   524,288       0.5     2     True         torch.float32
backbone/layer3/0/downsample/1/weight   [1024]              1,024         0       0     True         torch.float32
backbone/layer3/0/downsample/1/bias     [1024]              1,024         0       0     True         torch.float32
backbone/layer3/1/conv1/weight          [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/1/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/1/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/1/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/1/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/1/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/1/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/1/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/1/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/2/conv1/weight          [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/2/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/2/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/2/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/2/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/2/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/2/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/2/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/2/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/3/conv1/weight          [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/3/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/3/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/3/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/3/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/3/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/3/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/3/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/3/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/4/conv1/weight          [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/4/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/4/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/4/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/4/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/4/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/4/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/4/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/4/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/5/conv1/weight          [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/5/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/5/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/5/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/5/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/5/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/5/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/5/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/5/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/6/conv1/weight          [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/6/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/6/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/6/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/6/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/6/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/6/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/6/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/6/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/7/conv1/weight          [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/7/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/7/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/7/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/7/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/7/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/7/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/7/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/7/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/8/conv1/weight          [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/8/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/8/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/8/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/8/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/8/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/8/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/8/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/8/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/9/conv1/weight          [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/9/bn1/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/9/bn1/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/9/conv2/weight          [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/9/bn2/weight            [256]               256           0       0     True         torch.float32
backbone/layer3/9/bn2/bias              [256]               256           0       0     True         torch.float32
backbone/layer3/9/conv3/weight          [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/9/bn3/weight            [1024]              1,024         0       0     True         torch.float32
backbone/layer3/9/bn3/bias              [1024]              1,024         0       0     True         torch.float32
backbone/layer3/10/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/10/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/10/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/10/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/10/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/10/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/10/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/10/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/10/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/11/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/11/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/11/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/11/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/11/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/11/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/11/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/11/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/11/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/12/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/12/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/12/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/12/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/12/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/12/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/12/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/12/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/12/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/13/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/13/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/13/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/13/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/13/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/13/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/13/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/13/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/13/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/14/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/14/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/14/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/14/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/14/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/14/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/14/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/14/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/14/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/15/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/15/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/15/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/15/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/15/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/15/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/15/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/15/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/15/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/16/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/16/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/16/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/16/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/16/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/16/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/16/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/16/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/16/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/17/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/17/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/17/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/17/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/17/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/17/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/17/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/17/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/17/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/18/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/18/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/18/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/18/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/18/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/18/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/18/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/18/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/18/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/19/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/19/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/19/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/19/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/19/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/19/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/19/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/19/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/19/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/20/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/20/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/20/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/20/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/20/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/20/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/20/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/20/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/20/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/21/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/21/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/21/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/21/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/21/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/21/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/21/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/21/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/21/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer3/22/conv1/weight         [256, 1024, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/22/bn1/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/22/bn1/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/22/conv2/weight         [256, 256, 3, 3]    589,824       0.56    2.25  True         torch.float32
backbone/layer3/22/bn2/weight           [256]               256           0       0     True         torch.float32
backbone/layer3/22/bn2/bias             [256]               256           0       0     True         torch.float32
backbone/layer3/22/conv3/weight         [1024, 256, 1, 1]   262,144       0.25    1     True         torch.float32
backbone/layer3/22/bn3/weight           [1024]              1,024         0       0     True         torch.float32
backbone/layer3/22/bn3/bias             [1024]              1,024         0       0     True         torch.float32
backbone/layer4/0/conv1/weight          [512, 1024, 1, 1]   524,288       0.5     2     True         torch.float32
backbone/layer4/0/bn1/weight            [512]               512           0       0     True         torch.float32
backbone/layer4/0/bn1/bias              [512]               512           0       0     True         torch.float32
backbone/layer4/0/conv2/weight          [512, 512, 3, 3]    2,359,296     2.25    9     True         torch.float32
backbone/layer4/0/bn2/weight            [512]               512           0       0     True         torch.float32
backbone/layer4/0/bn2/bias              [512]               512           0       0     True         torch.float32
backbone/layer4/0/conv3/weight          [2048, 512, 1, 1]   1,048,576     1       4     True         torch.float32
backbone/layer4/0/bn3/weight            [2048]              2,048         0       0.01  True         torch.float32
backbone/layer4/0/bn3/bias              [2048]              2,048         0       0.01  True         torch.float32
backbone/layer4/0/downsample/0/weight   [2048, 1024, 1, 1]  2,097,152     2       8     True         torch.float32
backbone/layer4/0/downsample/1/weight   [2048]              2,048         0       0.01  True         torch.float32
backbone/layer4/0/downsample/1/bias     [2048]              2,048         0       0.01  True         torch.float32
backbone/layer4/1/conv1/weight          [512, 2048, 1, 1]   1,048,576     1       4     True         torch.float32
backbone/layer4/1/bn1/weight            [512]               512           0       0     True         torch.float32
backbone/layer4/1/bn1/bias              [512]               512           0       0     True         torch.float32
backbone/layer4/1/conv2/weight          [512, 512, 3, 3]    2,359,296     2.25    9     True         torch.float32
backbone/layer4/1/bn2/weight            [512]               512           0       0     True         torch.float32
backbone/layer4/1/bn2/bias              [512]               512           0       0     True         torch.float32
backbone/layer4/1/conv3/weight          [2048, 512, 1, 1]   1,048,576     1       4     True         torch.float32
backbone/layer4/1/bn3/weight            [2048]              2,048         0       0.01  True         torch.float32
backbone/layer4/1/bn3/bias              [2048]              2,048         0       0.01  True         torch.float32
backbone/layer4/2/conv1/weight          [512, 2048, 1, 1]   1,048,576     1       4     True         torch.float32
backbone/layer4/2/bn1/weight            [512]               512           0       0     True         torch.float32
backbone/layer4/2/bn1/bias              [512]               512           0       0     True         torch.float32
backbone/layer4/2/conv2/weight          [512, 512, 3, 3]    2,359,296     2.25    9     True         torch.float32
backbone/layer4/2/bn2/weight            [512]               512           0       0     True         torch.float32
backbone/layer4/2/bn2/bias              [512]               512           0       0     True         torch.float32
backbone/layer4/2/conv3/weight          [2048, 512, 1, 1]   1,048,576     1       4     True         torch.float32
backbone/layer4/2/bn3/weight            [2048]              2,048         0       0.01  True         torch.float32
backbone/layer4/2/bn3/bias              [2048]              2,048         0       0.01  True         torch.float32
