import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score





all_label=np.load('conc_CNN_label.npy')

test_label=all_label[0:31249,]
train_label=all_label[31250:231251,]

sums_train=[]
for i in range(train_label.shape[1]):
	sums_train.append(sum(train_label[:,i]))

sums_test=[]
for i in range(test_label.shape[1]):
	sums_test.append(sum(test_label[:,i]))


ind=np.arange(9)	
width=0.15
fig, ax = plt.subplots()
rects1 = ax.bar(ind, sums_test, width, color='b')
ax.set_ylabel('Number of positive samples')
ax.set_title('Distribution of positive samples across 9 labels in testing data')
ax.set_xticks(ind + width)
ax.set_xticklabels(('good_for_lunch', 'good_for_dinner', 'takes_res', 'outdoor_seat', 'rest_is_expensive','has_alcohol','has_table_service','amb_is_classy','good_for_kids'))
plt.show()


fig, ax = plt.subplots()
rects1 = ax.bar(ind, sums_train, width, color='b')
ax.set_ylabel('Number of positive samples')
ax.set_title('Distribution of positive samples across 9 labels in training data')
ax.set_xticks(ind + width)
ax.set_xticklabels(('good_for_lunch', 'good_for_dinner', 'takes_res', 'outdoor_seat', 'rest_is_expensive','has_alcohol','has_table_service','amb_is_classy','good_for_kids'))
plt.show()




#To plot F1/precision/recall graph 


f1_cal_predicted=np.load('files_for_f1_calc/predicted_class_0.npy')
f1_cal_true=np.load('files_for_f1_calc/truth_class_0.npy')
rows=f1_cal_predicted.shape[0]
f1_cal_predicted=np.reshape(f1_cal_predicted,(rows,1))
f1_cal_true=np.reshape(f1_cal_true,(rows,1))


for k in range(1,9):
	f1_cal_predicted=np.hstack((f1_cal_predicted,np.reshape(np.load('files_for_f1_calc/predicted_class_'+str(k)+'.npy'),(rows,1))))
	f1_cal_true=np.hstack((f1_cal_true,np.reshape(np.load('files_for_f1_calc/truth_class_'+str(k)+'.npy'),(rows,1))))


# F1 score for each label
f1_each_label=[]
prec_each_label=[]
recall_each_label=[]
for k in range(f1_cal_predicted.shape[1]):
	f1_each_label.append(f1_score(f1_cal_true[:,k],f1_cal_predicted[:,k]))
	prec_each_label.append(precision_score(f1_cal_true[:,k],f1_cal_predicted[:,k]))
	recall_each_label.append(recall_score(f1_cal_true[:,k],f1_cal_predicted[:,k]))
print "F1-score for each label "
print f1_each_label	
print "Precision for each label"
print prec_each_label
print "Recall for each label"
print recall_each_label

width=0.2
ind=np.arange(9)
fig, ax = plt.subplots()
rects1 = ax.bar(ind, f1_each_label, width, color='b')
rects2= ax.bar(ind+width,prec_each_label,width,color='g')
rects3=ax.bar(ind+2*width,recall_each_label,width,color='r')
ax.set_ylabel('Scores')
ax.set_title('F1-score/Precision/Recall')
ax.set_xticks(ind + width)
ax.set_xticklabels(('good_for_lunch', 'good_for_dinner', 'takes_res', 'outdoor_seat', 'rest_is_expensive','has_alcohol','has_table_service','amb_is_classy','good_for_kids'))
ax.legend((rects1[0], rects2[0],rects3[0]), ('F1-score', 'Precision','Recall'))
plt.show()
#fig.savefig('F1_precision_recall.jpg')

# F1 score for all the labels combined
f1=[]
prec=[]
rec=[]
for i in range(f1_cal_predicted.shape[0]):
	f1.append(f1_score(f1_cal_true[i,],f1_cal_predicted[i,]))
	prec.append(precision_score(f1_cal_true[i,],f1_cal_predicted[i,]))
	rec.append(recall_score(f1_cal_true[i,],f1_cal_predicted[i,]))
print "Mean F1-score"
print np.mean(f1)
print "Mean Precision score"
print np.mean(prec)
print "Mean recall score"
print np.mean(rec)






#sample Code for BAR GRAPHS
'''


fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)

womenMeans = (25, 32, 34, 20, 25)
womenStd = (3, 5, 2, 3, 3)
rects2 = ax.bar(ind + width, womenMeans, width, color='y', yerr=womenStd)

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()
'''