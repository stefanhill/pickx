from pickX.utils.ExecutionHandler import ExecutionHandler


training_path = "C:/git/pickx/data/Testing"
testing_path = "C:/git/pickx/data/Hessigheim"
#ExecutionHandler.prepare_dataset_for_training(training_path)

ExecutionHandler.train('model1', training_path, (([[128, 5], [64, 3]], 2, 32), 10))
evaluation = ExecutionHandler.test('model1', testing_path)