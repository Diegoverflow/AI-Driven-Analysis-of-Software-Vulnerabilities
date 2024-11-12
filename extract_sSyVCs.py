import clang.cindex
import pickle

DANGEROUS_FUNCS = ["func", "printf", "inner_function"] #strcpy

def parse_source_file(source_code):
    clang.cindex.Config.set_library_file(
        '/usr/lib/llvm-18/lib/libclang-18.so.18')
    index = clang.cindex.Index.create()
    tu = index.parse(source_code)
    return tu



def check_library_function_calls(node, sSyVCs):
    """Check if the node is a function call (Library/API Function Call)."""
    if node.spelling in DANGEROUS_FUNCS:
        if node.spelling not in sSyVCs["funcs"]:
            print('PRINTO ARGOMENTI FUNZIONE ' + node.spelling)
            possible_arguments = [pross_arg[1] for pross_arg in sSyVCs["others"]]
            for arg in node.get_arguments():
                if arg.spelling in possible_arguments:
                    print(arg.spelling)
                    print('adding ' + node.spelling + 'in check_library_function_calls')
                    #sSyVCs.append(node.spelling)
                    sSyVCs_funcs = sSyVCs["funcs"]
                    sSyVCs_funcs.append(node.spelling)
                    break
            print('FINE PRINT ARGOMENTI FUNZIONE ' + node.spelling)



def check_array_definitions(node, sSyVCs):
    """Check if the node is an array definition."""
    if node.kind == clang.cindex.CursorKind.VAR_DECL:
        if node.type.kind == clang.cindex.TypeKind.CONSTANTARRAY \
                or node.type.kind == clang.cindex.TypeKind.INCOMPLETEARRAY \
                or node.type.kind == clang.cindex.TypeKind.VARIABLEARRAY \
                or node.type.kind == clang.cindex.TypeKind.DEPENDENTSIZEDARRAY:
            if node.spelling not in sSyVCs:
                print('adding ' + node.spelling + 'in check_array_definitions')
                print('line ' + str(node.location.line))
                #sSyVCs.append(node.spelling)
                sSyVCs_others = sSyVCs["others"]
                array_definition = (node.location.line, node.spelling)
                sSyVCs_others.append(array_definition)


def check_pointer_definitions(node, sSyVCs):
    if node.kind == clang.cindex.CursorKind.VAR_DECL:
        if node.type.kind == clang.cindex.TypeKind.POINTER \
                or node.type.kind == clang.cindex.TypeKind.MEMBERPOINTER:
            if node.spelling not in sSyVCs:
                print('adding ' + node.spelling + 'in check_pointer_definitions')
                #sSyVCs.append(node.spelling)
                sSyVCs_others = sSyVCs["others"]
                pointer_definition = (node.location.line, node.spelling)
                sSyVCs_others.append(pointer_definition)


def check_arithmetic_expressions(node, sSyVCs):
    if node.kind == clang.cindex.CursorKind.VAR_DECL:
        for child in node.get_children():
            if child.kind == clang.cindex.CursorKind.BINARY_OPERATOR \
                    or child.kind == clang.cindex.CursorKind.UNARY_OPERATOR:
                if node.spelling not in sSyVCs:
                    print('adding ' + node.spelling + 'in check_arithmetic_expressions')
                    #sSyVCs.append(node.spelling)
                    sSyVCs_others = sSyVCs["others"]
                    arithmetic_expression_definition = (node.location.line, node.spelling)
                    sSyVCs_others.append(arithmetic_expression_definition)


def traverse_ast(node, sSyVCs):
    """Traverse the AST and apply checks."""
    check_library_function_calls(node, sSyVCs)
    check_array_definitions(node, sSyVCs)
    check_pointer_definitions(node, sSyVCs)
    check_arithmetic_expressions(node, sSyVCs)

    for child in node.get_children():
        traverse_ast(child, sSyVCs)


def main(file_path):
    """Main function to parse the file and traverse the AST."""
    translation_unit = parse_source_file(file_path)
    root_node = translation_unit.cursor
    sSyVCs = dict()
    sSyVCs["funcs"] = []
    sSyVCs["others"] = []
    traverse_ast(root_node, sSyVCs)
    sSyVCs_funcs = sSyVCs["funcs"]
    print("FUNCS")
    print(sSyVCs_funcs.__str__())
    sSyVCs_others = sSyVCs["others"]
    print("OTHERS")
    for tuple in sSyVCs_others:
        print(str(tuple))
    file_path = '/home/httpiego/PycharmProjects/VulDeeDiegator/TestPrograms/source/sSyVCs'
    with open(file_path, 'wb') as f:
        pickle.dump(sSyVCs, f)


if __name__ == "__main__":
    source_file = "/home/httpiego/PycharmProjects/VulDeeDiegator/TestPrograms/source/source.c"
    main(source_file)
