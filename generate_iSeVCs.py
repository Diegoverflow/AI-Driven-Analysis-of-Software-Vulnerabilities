import subprocess
import re


def create_path(path):
    clang_command = ['mkdir', path]

    try:
        # Run the clang command
        subprocess.run(clang_command, check=True)
        print(f"path has been successfully created")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while generating the path: {e}")


def clean(input_file, output_file, function_name):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    in_function = False
    function_body = []

    for line in lines:
        if re.match(rf'^define.*@{function_name}\(', line.strip()):
            in_function = True
        # If inside the function, append the line to the function body
        if in_function:
            function_body.append(line)
            # Detect the end of the function
            if line.strip() == '}':
                break

    # Write the extracted function body to the output file
    with open(output_file, 'w') as file:
        for line in function_body:
            file.write(line)


def slice(c_source_file, path, criteria):
    output_llvm_file = path + '/llvm.bc'
    generate_llvm = ['clang', '-g', '-S', '-emit-llvm', c_source_file, '-o', output_llvm_file]
    #generate_llvm = ['clang', '-g', '-c', '-emit-llvm', c_source_file, '-o', output_ir_file]

    sliced_file = path + '/example.sliced'
    #slice_llvm = ['/home/httpiego/dg/tools/llvm-slicer', '--preserve-dbg=false', '-c', '11:my_array', ir_file, '-o', sliced_file]
    slice_llvm = ['/home/httpiego/dg/tools/llvm-slicer',
                  '--forward',
                  '--preserve-dbg=false',
                  '-c', criteria,
                  output_llvm_file, '-o', sliced_file]

    sliced_file_no_dbg = path + '/no_dbg.sliced'
    delete_dbg_info = ['opt',
                       '--strip-debug',
                       '--strip-named-metadata',
                       '-dce',
                       sliced_file,
                       '-o', sliced_file_no_dbg]

    readable_no_dbg = path + '/no_dbg.readable'
    convert_no_dbg_2readable = ['llvm-dis', sliced_file_no_dbg, '-o', readable_no_dbg]

    #delete_dbg = ['llvm-strip', readable_sliced, '-o', sliced_file_no_dbg]

    readable_2c = path + '/sliced.c'
    convert_readable_2c = ['/home/httpiego/llvm2c/build/llvm2c', readable_no_dbg, '-o', readable_2c]

    cleaned_readable = path + '/cleaned.readable'

    try:
        # Run the clang command
        subprocess.run(generate_llvm, check=True)
        subprocess.run(slice_llvm, check=True)
        subprocess.run(delete_dbg_info, check=True)
        subprocess.run(convert_no_dbg_2readable, check=True)
        clean(readable_no_dbg, cleaned_readable, 'main')
        subprocess.run(convert_readable_2c, check=True)
        print(f"slicing done")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while slicing: {e}")

    return cleaned_readable


def find_called_functions_in_llvm(file_content):  #llvm_file_path):
    try:
        #with open(file_content, 'r+') as file:
        #    llvm_content = file.read()

        to_single_string = "".join(file_content)

        # Regular expression to find all function calls in the LLVM IR
        pattern = re.compile(r'\bcall\b.*@(\w+)\s*\(')

        matches = pattern.findall(to_single_string)
        # Removing duplicates and maintaining order
        called_functions = list(dict.fromkeys(matches))

        #for added_func in added_funcs:
        #    called_functions.remove(added_func)

        return called_functions

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []


def extract_func(input_file, function_name):
    #with open(input_file, 'r') as file:
    #    lines = file.readlines()

    in_function = False
    function_body = []

    for line in input_file:  #lines:
        if re.match(rf'^define.*@{function_name}\(', line.strip()):
            in_function = True
        # If inside the function, append the line to the function body
        if in_function:
            function_body.append(line)
            # Detect the end of the function
            if line.strip() == '}':
                break

    return function_body


def infite_loop(added_funcs: list, next_func):
    if len(added_funcs) <= 3:
        return False
    else:

        if (added_funcs[-3] == added_funcs[-1]
                and added_funcs[-2] == next_func):
            return True
        else:
            return False


def append_functions(llvm, program_slice, added_funcs: list):
    called_functions = find_called_functions_in_llvm(program_slice)
    for func in called_functions:

        if infite_loop(added_funcs, func):
            continue
        else:
            added_funcs.append(func)

            func_body = extract_func(llvm, func)

            updated_program_slice = []

            for line in program_slice:  # lines:
                updated_program_slice.append(line)
                #if re.match(r'\bcall\b.*\b' + re.escape(func) + r'\b', line.strip()):
                if f"call " in line and f"@{func}(" in line:
                    func_body = append_functions(llvm, func_body, added_funcs)
                    for func_body_line in func_body:
                        updated_program_slice.append(func_body_line)

            program_slice = updated_program_slice

            added_funcs = []

    #print(str(cleaned_readable))
    return program_slice


def find_max_var(iSeVC):

    max_var = -1

    var_pattern = r"%(\d+)"

    for line in iSeVC:
        match = re.search(var_pattern, line)
        if match:
            found_var = int(match.group(1))
            if found_var > max_var:
                max_var = found_var

    return max_var


def fix_variables(iSeVC, starting_variable):

    updated_iSeVC = []

    max_var = starting_variable

    all_reassigned_variables = []

    vars_dict_in_use = -1

    i = 0
    while i < len(iSeVC):
        line = iSeVC[i]
        if ':' in line:
            for j in range(len(line)):
                if line[j] == ':':
                        loop_variable = ''
                        while str(line[j-1]).isdigit():
                            loop_variable += line[j - 1]
                            line = line[:j-1] + line[j:]
                        loop_variable = loop_variable[::-1]
                        reassigned_variables = dict(all_reassigned_variables[vars_dict_in_use])
                        line = line[:j] + reassigned_variables[loop_variable] + line[j:]
                        #line = reassigned_variables[loop_variable] + line[j:]
                        break
        if '%' in line and vars_dict_in_use != -1:
            for j in range(len(line)):
                if line[j] == '%':
                        reassigned_variable = ''
                        while str(line[j + 1]).isdigit():
                            reassigned_variable += line[j + 1]
                            line = line[:j + 1] + line[j + 2:]
                        reassigned_variables = dict(all_reassigned_variables[vars_dict_in_use])
                        if reassigned_variables.get(reassigned_variable) is not None:
                            line = line[:j + 1] + reassigned_variables[reassigned_variable] + line[j + 1:]
                        else:
                            max_var += 1
                            line = line[:j + 1] + str(max_var) + line[j+1:]
                            reassigned_variables[reassigned_variable] = str(max_var)
                            all_reassigned_variables[vars_dict_in_use] = reassigned_variables
            #iSeVC[i] = line
            #updated_iSeVC.append(line)
        if i+1 < len(iSeVC):
            next_line = iSeVC[i+1]
            if 'call' in line and 'define' in next_line:
                func_args_to_be_changed = []
                reassigned_variables = {}
                vars_dict_in_use += 1
                for j in range(len(line)):
                    if line[j] == '%':
                        var = ''
                        while str(line[j+1]).isdigit():
                            var += line[j+1]
                            j += 1
                        func_args_to_be_changed.append(var)
                arg_changed = 0
                for k in range(len(next_line)):
                    if next_line[k] == '%':
                        reassigned_variable = ''
                        while str(next_line[k+1]).isdigit():
                            reassigned_variable += next_line[k+1]
                            next_line = next_line[:k+1] + next_line[k+2:]
                        next_line = next_line[:k+1] + func_args_to_be_changed[arg_changed] + next_line[k+1:]
                        reassigned_variables[reassigned_variable] = func_args_to_be_changed[arg_changed]
                        arg_changed += 1
                        #k += len(func_args_to_be_changed[arg_changed])
                all_reassigned_variables.append(reassigned_variables)
                iSeVC[i] = line
                updated_iSeVC.append(line)
                iSeVC[i+1] = next_line
                updated_iSeVC.append(next_line)
                i = i + 2
                continue
        if '}' in line:
            #all_reassigned_variables = all_reassigned_variables.pop(len(all_reassigned_variables)-1)
            #all_reassigned_variables.pop(len(all_reassigned_variables)-1)
            all_reassigned_variables = all_reassigned_variables[:-1]
            vars_dict_in_use -= 1
            updated_iSeVC.append(line)
            i = i + 1
            continue

        iSeVC[i] = line
        updated_iSeVC.append(line)
        i = i + 1

    return updated_iSeVC


# def fix_variables(iSeVC, starting_variable):
#
#     updated_iSeVC = []
#
#     max_var = starting_variable
#
#     all_reassigned_variables = []
#
#     vars_dict_in_use = -1
#
#     i = 0
#     while i < len(iSeVC):
#         line = iSeVC[i]
#         if i+1 < len(iSeVC):
#             next_line = iSeVC[i+1]
#             if 'call' in line and 'define' in next_line:
#                 func_args_to_be_changed = []
#                 reassigned_variables = {}
#                 vars_dict_in_use += 1
#                 for j in range(len(line)):
#                     if line[j] == '%':
#                         var = ''
#                         while str(line[j+1]).isdigit():
#                             var += line[j+1]
#                             j += 1
#                         func_args_to_be_changed.append(var)
#                 arg_changed = 0
#                 for k in range(len(next_line)):
#                     if next_line[k] == '%':
#                         reassigned_variable = ''
#                         while str(next_line[k+1]).isdigit():
#                             reassigned_variable += next_line[k+1]
#                             next_line = next_line[:k+1] + next_line[k+2:]
#                         next_line = next_line[:k+1] + func_args_to_be_changed[arg_changed] + next_line[k+1:]
#                         reassigned_variables[reassigned_variable] = func_args_to_be_changed[arg_changed]
#                         arg_changed += 1
#                         #k += len(func_args_to_be_changed[arg_changed])
#                 all_reassigned_variables.append(reassigned_variables)
#                 updated_iSeVC.append(line)
#                 updated_iSeVC.append(next_line)
#                 i = i + 2
#                 continue
#         if '%' in line and vars_dict_in_use != -1:
#             for j in range(len(line)):
#                 if line[j] == '%':
#                     if j == 2:
#                         reassigned_variable = ''
#                         while str(line[j + 1]).isdigit():
#                             reassigned_variable += line[j + 1]
#                             line = line[:j + 1] + line[j + 2:]
#                         line = line[:j + 1] + str(max_var) + line[j+1:]
#                         max_var += 1
#                         reassigned_variables = dict(all_reassigned_variables[vars_dict_in_use])
#                         reassigned_variables[reassigned_variable] = str(max_var)
#                         all_reassigned_variables[vars_dict_in_use] = reassigned_variables
#                         #j += len(str(max_var))
#                     else:
#                         if (j - 6) >= 0:
#                             if line[j-6] == 'l' and line[j-5] == 'a' and line[j-4] == 'b' and line[j-3] == 'e' and line[j-2] == 'l':
#                                 reassigned_variable = ''
#                                 while str(line[j + 1]).isdigit():
#                                     reassigned_variable += line[j + 1]
#                                     line = line[:j + 1] + line[j + 2:]
#                                 line = line[:j + 1] + str(max_var) + line[j + 1:]
#                                 max_var += 1
#                                 reassigned_variables = dict(all_reassigned_variables[vars_dict_in_use])
#                                 reassigned_variables[reassigned_variable] = str(max_var)
#                                 all_reassigned_variables[vars_dict_in_use] = reassigned_variables
#                                 continue
#                         reassigned_variable = ''
#                         while str(line[j + 1]).isdigit():
#                             reassigned_variable += line[j + 1]
#                             line = line[:j + 1] + line[j + 2:]
#                         reassigned_variables = dict(all_reassigned_variables[vars_dict_in_use])
#                         #for key in reassigned_variables:
#                         #    print(str(key) + ': ' + str(reassigned_variables[key]))
#                         #print(str(reassigned_variables[reassigned_variable]))
#                         line = line[:j+1] + reassigned_variables[reassigned_variable] + line[j+1:]
#             updated_iSeVC.append(line)
#             i = i + 1
#             continue
#         if '}' in line:
#             all_reassigned_variables = all_reassigned_variables.pop()
#             vars_dict_in_use -= 1
#             updated_iSeVC.append(line)
#             i = i + 1
#             continue
#         else:
#             updated_iSeVC.append(line)
#             i = i + 1
#
#     return updated_iSeVC



def main():

    sSyVCs = {}

    #foreach candidate in sSyVCs
    criteria = 'inner_function'
    c_source_file = '/home/httpiego/PycharmProjects/VulDeeDiegator/TestPrograms/source.c'

    path = '/home/httpiego/PycharmProjects/VulDeeDiegator/TestPrograms/' + criteria

    create_path(path)

    program_slice = slice(c_source_file, path, criteria)

    llvm_file = '/home/httpiego/PycharmProjects/VulDeeDiegator/TestPrograms/prova'
    generate_llvm = ['clang', '-S', '-emit-llvm', c_source_file, '-o', llvm_file]
    subprocess.run(generate_llvm, check=True)

    with open(llvm_file, 'r') as llvm:
        llvm_content = llvm.readlines()

    with open(program_slice, 'r') as ps:
        program_slice_content = ps.readlines()

    iSeVC = append_functions(llvm_content, program_slice_content, list())

    for line in iSeVC:
        print(str(line))

    print('_____________________________________________________________________________________')

    starting_variable = find_max_var(program_slice_content)

    final_iSeVC = fix_variables(iSeVC, starting_variable)

    for line in final_iSeVC:
        print(str(line))

if __name__ == "__main__":
    main()
