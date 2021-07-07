echo ''
echo ''
echo '############# Series1 - drift detection no update #############'
python3 ./concept_drift_detector.py ./data/series1.csv ./output/series1_no_update.csv \
1000 0 0.005 100 1

echo ''
echo ''
echo '############# Series1 - drift detection with update #############'
python3 ./concept_drift_detector.py ./data/series1.csv ./output/series1_with_update.csv \
1000 1 0.005 100 1

echo ''
echo ''
echo '############# Series1 - distribution estimation #############'
python3 ./distribution_estimator.py ./data/series1.csv ./output/series1_dist.csv \
1000 1

echo ''
echo ''
echo '############# Series2 - drift detection no update #############'
python3 ./concept_drift_detector.py ./data/series2.csv ./output/series2_no_update.csv \
1000 0 0.005 100 1

echo ''
echo ''
echo '############# Series2 - drift detection with update #############'
python3 ./concept_drift_detector.py ./data/series2.csv ./output/series2_with_update.csv \
1000 1 0.005 100 1

echo ''
echo ''
echo '############# Series2 - distribution estimation #############'
python3 ./distribution_estimator.py ./data/series2.csv ./output/series2_dist.csv \
1000 1

echo ''
echo ''
echo '############# Series3 - drift detection no update #############'
python3 ./concept_drift_detector.py ./data/series3.csv ./output/series3_no_update.csv \
1000 0 0.005 100 1

echo ''
echo ''
echo '############# Series3 - with update #############'
python3 ./concept_drift_detector.py ./data/series3.csv ./output/series3_with_update.csv \
1000 1 0.005 100 1

echo ''
echo ''
echo '############# Series3 - distribution estimation #############'
python3 ./distribution_estimator.py ./data/series3.csv ./output/series3_dist.csv \
1000 1
