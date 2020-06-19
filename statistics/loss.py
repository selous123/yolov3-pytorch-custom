
## 计算 YoLo-v3 的损失
def compute_loss(p, targets, model):  # predictions, targets, model

    ### targets: Tensor type with shapes=>(n_obj, 6) = (images_index, cls, gt_box[xywh])
    ### predictions: List type len(prediction) = 3 =>[tensor1, tensor2, tensor3]
    ###[感受野大][大] tensor1 with shape (16, 3, 16, 16, 49) => 16 batch_size + 3 anchors + 16x16 feature map + 49 (1 + 4 + 44)
    ###         [中] tensor2 with shape (16, 3, 32, 32, 49) =>
    ###         [小] tensor3 with shape (16, 3, 64, 64, 49) =>
    ### model: 需要读取模型的配置文件

    ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor  ### 定义数据类型
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0]) ### 创建损失项, lcls:类别损失, lbox:回归损失, lobj: 类别损失
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # 返回 gtbox 与 哪些 anchor 匹配
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    nt = 0  # targets
    ## 对于每一层的输出
    for i, pi in enumerate(p):  # layer index, layer predictions
        ## 取出对应每层的 indicies.
        ### ==> 图像b, 与 anchor a 在 gj 和 gi 位置匹配
        b, a, gj, gi = indices[i]  # image_index, anchor, gridy, gridx

        ### 预测的这个框 是否包含物体
        tobj = torch.zeros_like(pi[..., 0])  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets

            ## 取出 对应的所有的 prediction.
            ## 16 batch_size + 3 anchors + 16x16 feature map + 49 (1 + 4 + 44)
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            pxy = torch.sigmoid(ps[:, 0:2])
            pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchors[i] ## 框应该偏移多少
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            ## 计算 pbox 框 与 tbox 之间的 GIOU
            ## tbox=> (0,0,w,h) =>w,h is in grid scale..
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
            lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss

            # Obj
            ## 根据 giou 判断 prediction框 是否包含 物体
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class loss
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn)  # targets
                t[range(nb), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']
    if red == 'sum':
        bs = tobj.shape[0]  # batch size
        g = 3.0  # loss gain
        lobj *= g / bs
        if nt:
            lcls *= g / nt / model.nc
            lbox *= g / nt

    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()


def build_targets(p, targets, model):
################################################################
###Inputs:
### targets: Tensor type with shapes=>(n_obj, 6) = (images_index, cls, gt_box[xywh])
### predictions: List type len(prediction) = 3 =>[tensor1, tensor2, tensor3]
###[感受野大][大物体] tensor1 with shape (16, 3, 16, 16, 49) => 16 batch_size + 3 anchors + 16x16 feature map + 49 (1 + 4 + 44)
###[感受野中][中物体] tensor2 with shape (16, 3, 32, 32, 49) =>
###[感受野小][小物体] tensor3 with shape (16, 3, 64, 64, 49) =>
### model: 需要读取模型的配置文件
###Outputs: mn represents match number.
###  tcls: list with 3 tensors => shape [mn]
###  tbox: list with 3 tensors => shape [mn x 4]
###  indices: list with 3 tuple =>  image_index, anchor, (gj, gi)相对坐标grid位置
###  anch: list with 3 tensors => shape [mn x 2]
#################################################################

    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    nt = targets.shape[0] ## objects 的个数
    tcls, tbox, indices, anch = [], [], [], []
    ### gain => Tensor with [1, 1, 1, 1, 1, 1]
    gain = torch.ones(6, device=targets.device)
    ### offset
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offset
    style = None
    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    ###model.yolo_layers => list 记录 model中 yolo layer 的位置: [89(大), 101(中), 113(小)]
    ### self.strides = [32, 16, 8]
    for i, j in enumerate(model.yolo_layers):
        ## anchor_vec == anchors(大anchor) / self.stride (32)
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain => [1, 1, 16, 16, 16, 16]
        na = anchors.shape[0]  # number of anchors => 3
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
        ## at with shape [3 x nt] and values are [0, 1, 2]

        # Match targets to anchors  ==> anchors(ori_img / stride) = targets (norm * [512 / stride](gain))
        a, t, offsets = [], targets * gain, 0
        if nt:
            # r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
            # j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            ## 计算 anchors 和 target 的匹配度
            ## j with shape [3 x nt], 储存值为True or False, 记录 target 是否与 anchor 匹配
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))

            ## a 变成一维 记录匹配anchor编号
            ## t 记录那些 target的匹配，与a位置对应
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            if style == 'rect2':
                g = 0.2  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

            elif style == 'rect4':
                g = 0.5  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
        if c.shape[0]:  # if any targets
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())
        exit(0)
    return tcls, tbox, indices, anch
