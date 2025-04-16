import os
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import sklearn.metrics as sm


'''
    学习率：https://blog.csdn.net/weixin_45209433/article/details/112324325
    
'''


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            with torch.no_grad():
                # feature.data.t_(), target.data.sub_(1)
                feature.t_(), target.sub_(1)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                if batch.batch_size == args.batch_size:
                    sys.stdout.write(
                        '\rEpoch[{}] - Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,
                                                                                             steps,
                                                                                             loss.item(),
                                                                                             train_acc,
                                                                                             corrects,
                                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                result = eval(dev_iter, model, args)  # dev_acc, y_true, y_pred
                if len(result) > 1:
                    dev_acc, y_true, y_pred = result[0], result[1], result[2]
                else:
                    dev_acc, y_true, y_pred = result, None, None

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}%.'.format(best_acc))
                        save(model, args.save_dir, 'best', steps)
                    if args.confusion_matrix:
                        print('Confusion matrix:')
                        print(confusion_matrix(y_true, y_pred))

                    if args.show_score:
                        print('Score:')
                        accuracy = sm.accuracy_score(y_true, y_pred)*100
                        precision = sm.precision_score(y_true, y_pred, average=args.average_type)*100
                        recall = sm.recall_score(y_true, y_pred, average=args.average_type)*100
                        f1_score = sm.f1_score(y_true, y_pred, average=args.average_type)*100
                        auc = sm.roc_auc_score(y_true, y_pred, average=args.average_type)*100
                        print('accuracy={:.2f}%.\n'
                              'precision={:.2f}%.\n'
                              'recall={:.2f}%.\n'
                              'f1_score={:.2f}%.\n'
                              'auc={:.2f}%.'.format(accuracy, precision, recall, f1_score, auc)
                              )
                    if args.show_report:
                        print('Report:')
                        print(sm.classification_report(y_true, y_pred))
                else:
                    if steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}%'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:

        feature, target = batch.text, batch.label
        with torch.no_grad():
            # feature.data.t_(), target.data.sub_(1)
            feature.t_(), target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
        y_pred = torch.max(logits, 1)[1].view(target.size()).data
        y_true = target.data
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    if args.confusion_matrix:
        return accuracy, y_true, y_pred
    else:
        return accuracy


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
