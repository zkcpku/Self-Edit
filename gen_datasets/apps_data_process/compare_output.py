def compare_output(output, ground_truth,debug=True):
    if custom_compare_(output, ground_truth):
        tmp_result = True
        return tmp_result
    # ground truth sequences are expressed as lists not tuples
    if isinstance(output, tuple):
        output = list(output)

    tmp_result = False
    try:
        tmp_result = (output == [ground_truth])
        if isinstance(ground_truth, list):
            tmp_result = tmp_result or (output == ground_truth)
            if isinstance(output[0], str):
                tmp_result = tmp_result or ([e.strip() for e in output] == ground_truth)
    except Exception as e:
        if debug:
            print(f"Failed check1 exception = {e}")
        pass
    
    if tmp_result == True:  
        return tmp_result

    # try one more time without \n
    if isinstance(ground_truth, list):
        for tmp_index, i in enumerate(ground_truth):
            ground_truth[tmp_index] = i.split("\n")
            ground_truth[tmp_index] = [x.strip() for x in ground_truth[tmp_index] if x]
    else:
        ground_truth = ground_truth.split("\n")
        ground_truth = list(filter(len, ground_truth))
        ground_truth = list(map(lambda x:x.strip(), ground_truth))

    try:
        tmp_result = (output == [ground_truth])
        if isinstance(ground_truth, list):
            tmp_result = tmp_result or (output == ground_truth)
    except Exception as e:
        if debug:
            print(f"Failed check2 exception = {e}")
        pass
    
    
    if tmp_result == True:
        return tmp_result

    # try by converting the output into a split up list too
    if isinstance(output, list):
        output = list(filter(len, output))

    if debug:
        nl = "\n"
        if not isinstance(inputs, list):
            print(f"output = {output}, test outputs = {ground_truth}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [ground_truth]}") 
        else:
            print(f"output = {output}, test outputs = {ground_truth}, inputs = {inputs}, {type(inputs)}, {output == [ground_truth]}") 

    if tmp_result == True:
        return tmp_result

    try:
        tmp_result = (output == [ground_truth])
        if isinstance(ground_truth, list):
            tmp_result = tmp_result or (output == ground_truth)
    except Exception as e:
        if debug:
            print(f"Failed check3 exception = {e}")
        pass

    try:
        output_float = [float(e) for e in output]
        gt_float = [float(e) for e in ground_truth]
        tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
    except Exception as e:
        pass
    try:
        if isinstance(output[0], list):
            output_float = [float(e) for e in output[0]]
            gt_float = [float(e) for e in ground_truth[0]]
            tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
    except Exception as e:
        pass
    

    if tmp_result == True:
        return tmp_result

    # try by converting the stuff into split up list
    if isinstance(ground_truth, list):
        for tmp_index, i in enumerate(ground_truth):
            ground_truth[tmp_index] = set(i.split())
    else:
        ground_truth = set(ground_truth.split())

    try:
        tmp_result = (output == ground_truth)
    except Exception as e:
        if debug:
            print(f"Failed check4 exception = {e}")

    if tmp_result == True:
        return tmp_result

    # try by converting the output into a split up list too
    print(output)
    if isinstance(output, list):
        for tmp_index, i in enumerate(output):
            output[tmp_index] = i.split()
        output = list(filter(len, output))
        for tmp_index, i in enumerate(output):
            output[tmp_index] = set(i)    
    else:
        output = output.split()
        output = list(filter(len, output))
        output = set(output)
    print(output)
    print(set(frozenset(s) for s in output))
    print(set(frozenset(s) for s in ground_truth))
    try:
        tmp_result = (set(frozenset(s) for s in output) == set(frozenset(s) for s in ground_truth))
    except Exception as e:
        if debug:
            print(f"Failed check5 exception = {e}")
    print(tmp_result)
    assert False
    # if they are all numbers, round so that similar numbers are treated as identical
    try:
        tmp_result = tmp_result or (set(frozenset(round(float(t),3) for t in s) for s in output) ==\
            set(frozenset(round(float(t),3) for t in s) for s in ground_truth))
    except Exception as e:
        if debug:
            print(f"Failed check6 exception = {e}")

    if tmp_result == True and debug:
        print("PASSED")

    if debug:
        nl = "\n"
        if not isinstance(inputs, list):
            print(f"output = {output}, test outputs = {ground_truth}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [ground_truth]}")
        else:
            print(f"output = {output}, test outputs = {ground_truth}, inputs = {inputs}, {type(inputs)}, {output == [ground_truth]}") 

            
    return tmp_result
    
    


def custom_compare_(output, ground_truth):
    
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False