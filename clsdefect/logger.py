import numpy as np
from clsdefect.transforms_image_mask_label import DecodeLabels
from clsdefect.utils import add_roc, add_prec_rec_curve, add_confusion_matrix


class Logger(object):
    def __init__(self, prefix, names, writer, loss=True, model=None, eps=1e-6):
        self.names = ["{}/{}".format(prefix, name) for name in names]
        self.classes = names
        self.pmfs = []
        self.predictions = []
        self.actuals = []
        self.writer = writer
        self.get_labels = DecodeLabels(len(names))
        self.loss_sum = 0
        self.loss_write = loss
        self.epsilon = eps  # for numerical stability
    #     if model is not None:
    #         self.actv_maps = []
    #         self.model = model
    #         self.model.hook_maps(self.hooker)
    # 
    # def hooker(self, module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
    #     return self.actv_maps.append(output.data.cpu().numpy())

    # @staticmethod
    # def returnCAM(feature_conv, weight_softmax, class_idx):
    #     # generate the class activation maps upsample to 256x256
    #     size_upsample = (256, 256)
    #     bz, nc, h, w = feature_conv.shape
    #     output_cam = []
    #     for idx in class_idx:
    #         cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
    #         cam = cam.reshape(h, w)
    #         cam = cam - np.min(cam)
    #         cam_img = cam / np.max(cam)
    #         cam_img = np.uint8(255 * cam_img)
    #         output_cam.append(cv2.resize(cam_img, size_upsample))
    #     return output_cam
    # 
    # def plt_CAMS(self, img):
    #     idx = range(len(self.classes))
    #     CAMs = self.returnCAM(self.actv_maps, self.model.weights_maps, idx)
    #     ch, height, width = img.shape
    #     results = []
    #     fig = plt.figure(figsize=(15, 6))
    #     for i, cam in enumerate(CAMs):
    #         heatmap = cv2.cvtColor(cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    #         result = heatmap * 0.3 + img * 0.5
    #         pil_img_cam = Image.fromarray(np.uint8(result))
    #         plt.subplot(1, 3, i + 1)
    #         plt.imshow(np.array(pil_img_cam))
    #         plt.title(self.classes[idx[i]])

    def append(self, pmfs, labels, loss=0):
        self.pmfs.extend(pmfs.detach().cpu().data.numpy())
        self.actuals.extend(labels.detach().cpu().data.numpy())
        if self.loss_write:
            self.loss_sum += loss.item()

    def write(self, epoch, th=None):
        # th = [0.35, 0.5, 0.05]
        if th is None:
            automatic_thresholding = True
        else:
            automatic_thresholding = False

        avg_balanced_accuracy = 0
        for i, name in enumerate(self.names):  # looping on different classes
            pmf = [e[i] for e in self.pmfs]
            true = [l[i] for l in self.actuals]
            eps = self.epsilon/len(pmf)
            _, _, _, auto_th, _ = add_roc(self.writer, epoch, true, pmf, name=f'{name}/ROC')
            # _, _, auto_th, _ = add_prec_rec_curve(self.writer, epoch, true, pmf, name=f'{name}/precision_recall')
            add_prec_rec_curve(self.writer, epoch, true, pmf, name=f'{name}/precision_recall')
            threshold = auto_th if automatic_thresholding else th[i]
            print(f'threshold is {threshold}')
            if eps == 0:
                eps = self.epsilon
            TP = sum([1 for p, l in zip(pmf, true) if p >= threshold and l == 1])/len(pmf)
            TN = sum([1 for p, l in zip(pmf, true) if p < threshold and l == 0])/len(pmf)
            FP = sum([1 for p, l in zip(pmf, true) if p >= threshold and l == 0])/len(pmf)
            FN = sum([1 for p, l in zip(pmf, true) if p < threshold and l == 1])/len(pmf)
            Correct = TN + TP
            LN = TN + FP  # Labeled Negative
            LP = TP + FN  # Labeled Positive
            PP = TP + FP  # Predicted Positive
            PN = TN + FN  # Predicted Negative
            Total = TN + TP + FP + FN
            accuracy = Correct / (Total + eps)
            recall = TP / (LP + eps)
            precision = TP / (PP + eps)
            selectivity = TN / (LN + eps)
            balanced_accuracy = 0.5*(recall + selectivity)
            avg_balanced_accuracy += balanced_accuracy
            f1 = 2*precision*recall / (precision + recall + eps)
            # noinspection PyCompatibility
            print(f"[{name}] TPR: {recall:.3f}, TNR: {selectivity:.3f}, Accuracy: {accuracy:.3f}/{balanced_accuracy:.3f}, "
                  f"Recall: {recall:.3f}, Precision: {precision:.3f}")
            self.writer.add_scalar("{}/true_positive".format(name), TP, epoch)
            self.writer.add_scalar("{}/true_negative".format(name), TN, epoch)
            self.writer.add_scalar("{}/false_positive".format(name), FP, epoch)
            self.writer.add_scalar("{}/false_negative".format(name), FN, epoch)
            self.writer.add_scalar("{}/accuracy".format(name), accuracy, epoch)
            self.writer.add_scalar("{}/balanced_accuracy".format(name), balanced_accuracy, epoch)
            self.writer.add_scalar("{}/selectivity".format(name), selectivity, epoch)
            self.writer.add_scalar("{}/precision".format(name), precision, epoch)
            self.writer.add_scalar("{}/recall".format(name), recall, epoch)
            self.writer.add_scalar("{}/f1".format(name), f1, epoch)
            add_confusion_matrix(self.writer, epoch, np.array([[selectivity, 1-selectivity], [1-recall, recall]]),
                                       labels=['(-)', '(+)'], name=f'{name}/confusion_matrix', normalize=False)

        avg_balanced_accuracy = avg_balanced_accuracy / len(self.names)
        self.writer.add_scalar("avg_balanced_accuracy", avg_balanced_accuracy, epoch)
        if self.loss_write:
            self.writer.add_scalar('train/loss', self.loss_sum, epoch)


class LoggerPercentage(object):
    def __init__(self, prefix, names, writer, loss=True, model=None, area_th=170, ratio_th=0.1, eps=1e-6):
        self.names = ["{}/{}".format(prefix, name) for name in names]
        self.classes = names
        self.get_labels = DecodeLabels(len(names))
        self.area_th = area_th
        self.ratio_th = ratio_th
        self.epsilon = eps  # for numerical stability
        self.true_area = [[] for _ in range(len(names))]
        self.pred_area = [[] for _ in range(len(names))]
        self.ship_area = []
        # self.true_area_overall = [[] for _ in range(len(names))]
        # self.pred_area_overall = [[] for _ in range(len(names))]
        self.pmfs = [[] for _ in range(len(names))]
        self.actuals = [[] for _ in range(len(names))]
        self.selectivity = [[] for _ in range(len(names))]
        self.recall = [[] for _ in range(len(names))]
        self.accuracy = [[] for _ in range(len(names))]
        self.balanced_accuracy = [[] for _ in range(len(names))]
        self.writer = writer

    def end_image(self, idx, th):
        avg_balanced_accuracy = 0
        for i, name in enumerate(self.names):  # looping on different classes
            # self.pred_area_overall[i].append(sum(self.pred_area[i]))
            # self.true_area_overall[i].append(sum(self.true_area[i]))
            pmf = self.pmfs[i]
            true = self.actuals[i]
            eps = self.epsilon / len(pmf)
            _, _, _, auto_th, _ = add_roc(self.writer, idx, true, pmf, name=f'{name}/ROC')
            # _, _, auto_th, _ = add_prec_rec_curve(self.writer, idx, true, pmf, name=f'{name}/precision_recall')
            add_prec_rec_curve(self.writer, idx, true, pmf, name=f'{name}/precision_recall')
            threshold = th[i]
            # print(f'threshold is {threshold}')
            if eps == 0:
                eps = self.epsilon
            TP = sum([1 for p, l in zip(pmf, true) if p >= threshold and l == 1]) / len(pmf)
            TN = sum([1 for p, l in zip(pmf, true) if p < threshold and l == 0]) / len(pmf)
            FP = sum([1 for p, l in zip(pmf, true) if p >= threshold and l == 0]) / len(pmf)
            FN = sum([1 for p, l in zip(pmf, true) if p < threshold and l == 1]) / len(pmf)
            Correct = TN + TP
            LN = TN + FP  # Labeled Negative
            LP = TP + FN  # Labeled Positive
            PP = TP + FP  # Predicted Positive
            PN = TN + FN  # Predicted Negative
            Total = TN + TP + FP + FN
            accuracy = Correct / (Total + eps)
            recall = TP / (LP + eps)
            precision = TP / (PP + eps)
            selectivity = TN / (LN + eps)
            balanced_accuracy = 0.5 * (recall + selectivity)
            avg_balanced_accuracy += balanced_accuracy
            f1 = 2 * precision * recall / (precision + recall + eps)
            # noinspection PyCompatibility
            print(
                f"[{name}] TPR: {recall:.3f}, TNR: {selectivity:.3f}, Accuracy: {accuracy:.3f}/{balanced_accuracy:.3f}, "
                f"Recall: {recall:.3f}, Precision: {precision:.3f}, Pred_Area: {self.pred_area[i][-1]: .4f}, "
                f"True_Area: {self.true_area[i][-1]: .4f}, Ship_Area: {self.ship_area[-1]: .4f}")
            self.writer.add_scalar("{}/true_positive".format(name), TP, idx)
            self.writer.add_scalar("{}/true_negative".format(name), TN, idx)
            self.writer.add_scalar("{}/false_positive".format(name), FP, idx)
            self.writer.add_scalar("{}/false_negative".format(name), FN, idx)
            self.writer.add_scalar("{}/accuracy".format(name), accuracy, idx)
            self.writer.add_scalar("{}/balanced_accuracy".format(name), balanced_accuracy, idx)
            self.writer.add_scalar("{}/selectivity".format(name), selectivity, idx)
            self.writer.add_scalar("{}/precision".format(name), precision, idx)
            self.writer.add_scalar("{}/recall".format(name), recall, idx)
            self.writer.add_scalar("{}/f1".format(name), f1, idx)
            self.writer.add_scalar("{}/pred_area".format(name), self.pred_area[i][-1], idx)
            self.writer.add_scalar("{}/true_area".format(name), self.true_area[i][-1], idx)
            self.writer.add_scalar("{}/pred_percentage".format(name), self.pred_area[i][-1]/self.ship_area[-1], idx)
            self.writer.add_scalar("{}/true_percentage".format(name), self.true_area[i][-1]/self.ship_area[-1], idx)
            self.writer.add_scalar("{}/ship_area".format(name), self.ship_area[-1], idx)
            self.selectivity[i].append(selectivity)
            self.recall[i].append(recall)
            self.accuracy[i].append(accuracy)
            self.balanced_accuracy[i].append(balanced_accuracy)
            add_confusion_matrix(self.writer, idx,
                                       np.array([[selectivity, 1 - selectivity], [1 - recall, recall]]),
                                       labels=['(-)', '(+)'], name=f'{name}/confusion_matrix', normalize=False)

        avg_balanced_accuracy = avg_balanced_accuracy / len(self.names)
        self.writer.add_scalar("avg_balanced_accuracy", avg_balanced_accuracy, idx)

    def end_all(self):
        for i, name in self.names:
            selectivity = sum(self.selectivity[i]) / len(self.selectivity[i])
            recall = sum(self.recall[i]) / len(self.recall[i])
            accuracy = sum(self.accuracy[i]) / len(self.accuracy[i])
            balanced_accuracy = sum(self.balanced_accuracy[i]) / len(self.balanced_accuracy[i])
            perc_pred_overall = sum(self.pred_area[i]) / len(self.pred_area[i])
            perc_true_overall = sum(self.true_area[i]) / len(self.true_area[i])
            self.writer.add_scalar("{}/accuracy".format(name), accuracy, -1)
            self.writer.add_scalar("{}/balanced_accuracy".format(name), balanced_accuracy, -1)
            self.writer.add_scalar("{}/pred_percentage".format(name), perc_pred_overall, -1)
            self.writer.add_scalar("{}/true_percentage".format(name), perc_true_overall, -1)
            add_confusion_matrix(self.writer, -1, np.array([[selectivity, 1 - selectivity], [1 - recall, recall]]),
                                       labels=['(-)', '(+)'], name=f'{name}/confusion_matrix', normalize=False)

    @staticmethod
    def get_area(pred, ship_mask):
        return pred.sum() / ship_mask.sum()

    def patch_write(self, pmfs, seg_mask, ship_mask, truth):
        if (ship_mask*seg_mask).sum() > self.area_th:
            pmfs = pmfs.squeeze().cpu().data.numpy()
            true = truth.mean((1, 2)).cpu().data.numpy() > self.ratio_th
            for i, name in enumerate(self.names):
                self.pmfs[i].append(pmfs[i])
                self.actuals[i].append(true[i])

    def image_write(self, idx, prediction, seg_mask, ship_mask, truth):
        predictions = [ship_mask * seg_mask * prediction[ic] for ic, cls in enumerate(self.classes)]
        truths = truth.split(1, 0)
        for ic, (cls, pred, tru) in enumerate(zip(self.classes, predictions, truths)):
            self.true_area[ic].append(tru.sum())
            self.pred_area[ic].append(pred.sum())
        self.ship_area.append(ship_mask.sum())
