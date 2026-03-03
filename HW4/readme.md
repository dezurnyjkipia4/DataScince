ДЗ №4

Датасет - кредитования. (https://www.kaggle.com/datasets/sujithmandala/simple-loan-classification-dataset)

Выполнено:

1.анализ датасета на наличие пропусков; 



2.приведение категориальных признаков датасета к числовым; 



3.удаление  параметров для повышения качества метрики (профессии заёмщиков) 



4.проверка данных на классификаторах:




  4.1 Классификатор градиентного бустинга


  
  4.2 Классификатор Extra Trees.


  
  4.3 Классификатор AdaBoost.
  4.4 Классификатор K-Nearest Neighbors.


  
  4.5 Классификатор дерева решений.


  
5. Визуализация



6.Заключение по классификаторам:
     Рейтинг моделей по точности:
1 1. ada_boosting: 1.0000    (переобучена при разных test_size, random_state)
2 2. DecisionTreeClassifier: 1.0000  (переобучена при разных test_size, random_state)
3 3. gradient_boosting: 0.9600
4 4. ExtraTreesClassifier: 0.9600
5 5. KNeighborsClassifier: 0.9600 
