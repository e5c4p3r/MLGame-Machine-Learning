# MLGame-Machine-Learning
訓練 AI 遊玩 MLGame 框架下的遊戲

## Arkanoid - 打磚塊遊戲

### 執行遊戲

ml_play_manual.py - 手動玩遊戲

ml_play_automatic.py - 自動玩遊戲 (預測球落點)

ml_play_model.py - AI 模型玩遊戲

執行指令：```python -m mlgame -f 60(幀率) -i ./ml/ml_play_manual(類型).py . --difficulty NORMAL(難度) --level 1(關卡)```

### 訓練模型

build_train_data.py - 蒐集訓練資料 (預設每關玩 5 次)

knn_train.py - 使用 KNN 訓練模型 (預設 k = 1)

play_level_model.py - 使用模型遊玩所有關卡
