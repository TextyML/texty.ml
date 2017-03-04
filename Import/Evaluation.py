import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import eli5
import pandas as pd

class Evaluation():
    
    def print_weights(self, clf, vecs, n = 50):
        feature_names = vecs[0].get_feature_names()
        for vec in vecs[1:]:
            feature_names.extend(vec.get_feature_names())
        
        return eli5.show_weights(clf, top=n, feature_names=feature_names)
    
    def print_probabilites(self, pipeline, text): 
        
        classes = pipeline.classes_
        probability = pipeline.predict_proba([text])
        
        matplotlib.style.use('ggplot')

        d = {'values': probability[0]}
        df = pd.DataFrame(d)


        ax = df.plot(kind='bar',legend=False,title='Predicted Probability of Input Text')
        ax.set_xticklabels(classes)
        ax.set_xlabel("Classes",fontsize=12)
        ax.set_ylabel("Probability",fontsize=12)

        plt.show()
    
    def print_report(self, pipe, X_test, Y_test):
        Y_pred = pipe.predict(X_test)
        report = metrics.classification_report(Y_test, Y_pred)
        print(report)
        print("accuracy: {:0.3f}".format(metrics.accuracy_score(Y_test, Y_pred)))
        np.set_printoptions(precision=2)
        classes = ["sport", "tech", "science", "politics", "opinion", "lifestyle", "economy", "business"]
        confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred, classes)
        plt.figure()
        self._plot_confusion_matrix(confusion_matrix,classes)
        plt.show()

    def _plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')