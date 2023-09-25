def simple_selection(input_data, output_data, proportion):
    new_input_data = []
    new_output_data = []
    new_input_test_data = []
    new_output_test_data = []
    total_items = len(input_data)
    count = 0
    for data, output in zip(input_data, output_data):
        if count / total_items <= proportion:
            new_input_data.append(data)
            new_output_data.append(output)
        else:
            new_input_test_data.append(data)
            new_output_test_data.append(output)
        count += 1

    return new_input_data, new_output_data, new_input_test_data, new_output_test_data


def k_fold_selection(input_data, output_data, folds):
    new_input_data = []
    new_output_data = []
    total_items = len(input_data)
    items_per_fold = total_items // folds
    reminding_items = total_items % folds


    #particionamos en K conjuntos
    current_data_idx = 0
    for fold_number in range(folds):
        new_input_data.append([])
        new_output_data.append([])
        current_inp_list = new_input_data[fold_number]
        current_output_list = new_output_data[fold_number]
        current_item = 0
        while current_item != items_per_fold :
            current_inp_list.append(input_data[current_data_idx])
            current_output_list.append(output_data[current_data_idx])
            current_data_idx += 1
            current_item += 1

    for fold_number_remaining in range(reminding_items):
        new_input_data[fold_number_remaining].append(input_data[current_data_idx])
        new_output_data[fold_number_remaining].append(output_data[current_data_idx])
        current_data_idx += 1

    return new_input_data, new_output_data
