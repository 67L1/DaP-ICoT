import json
import os


def sanitize_filename(filename):
    """
    Sanitizes a string to be used as a valid filename.
    Removes or replaces characters that are problematic in filenames.
    """
    # 允许字母、数字、下划线、连字符、点
    # 其他字符替换为空格，然后多个空格合并为一个，并去除首尾空格
    # 你可以根据需要调整这个替换逻辑
    filename = re.sub(r'[^\w\-\. ]', ' ', filename)  # 替换非法字符为空格
    filename = re.sub(r'\s+', ' ', filename).strip()  # 合并多余空格
    if not filename:  # 如果处理后为空，给个默认名
        filename = "unknown_category"
    return filename


def convert_jsonl_to_categorized_txt(jsonl_file_path, output_base_dir):
    """
    Converts a JSON Lines file to tab-separated values (TSV) format
    and saves data into different .txt files based on the 'category' field.
    Each output .txt file will be named after the category and placed in output_base_dir.

    Each line in the output .txt file will be:
    Image_Name + "\t" + Question + "\t" + Ground_Truth_Answer + "\t" + Your_Response + "\n"

    Args:
        jsonl_file_path (str): Path to the input JSON Lines file.
        output_base_dir (str): The base directory where category-specific .txt files will be saved.
    """
    # 用于存储已打开的文件句柄，键是 category 名称，值是文件对象
    open_files = {}

    try:
        # 确保输出目录存在
        os.makedirs(output_base_dir, exist_ok=True)

        with open(jsonl_file_path, 'r', encoding='utf-8') as infile:
            print(f"Processing input file: {jsonl_file_path}")
            for line_number, line in enumerate(infile, 1):
                try:
                    stripped_line = line.strip()
                    if not stripped_line:  # 跳过空行
                        # print(f"Skipping empty line at line {line_number}.")
                        continue

                    data = json.loads(stripped_line)

                    # 0. 获取 Category，并确定输出文件名
                    category_raw = data.get('category', 'unknown_category')
                    if not category_raw:  # 如果category是空字符串
                        category_raw = 'unknown_category'

                    # 对category名进行清理，使其成为合法的文件名
                    # 例如，替换掉路径分隔符等特殊字符
                    # category_filename = category_raw.replace('/', '_').replace('\\', '_') + ".txt"
                    # 使用更通用的清理函数
                    category_filename = sanitize_filename(category_raw) + ".txt"

                    output_txt_file_path = os.path.join(output_base_dir, category_filename)

                    # 1. 图片名称 (Image_Name)
                    image_name_raw = data.get('question_id', 'Unknown_Image_ID')
                    image_name = image_name_raw.replace("\n", " ").replace("\t", " ")

                    # 2. 问题 (Question)
                    question_raw = data.get('question', 'Unknown_Question')
                    question = question_raw.replace("\n", " ").replace("\t", " ")

                    # 3. 标准答案 (Ground_Truth_Answer)
                    ground_truth_answer_raw = data.get('answer', 'Unknown_GT_Answer')
                    ground_truth_answer = ground_truth_answer_raw.replace("\n", " ").replace("\t", " ")

                    # 4. 你的模型回答 (Your_Response)
                    your_response_raw = data.get('pred', 'No_Prediction_Available')
                    your_response_cleaned = your_response_raw.replace("\n", " ").replace("\t", " ")

                    # 检查是否有字段为空或为默认值
                    if image_name_raw == 'Unknown_Image_ID' or \
                            question_raw == 'Unknown_Question' or \
                            ground_truth_answer_raw == 'Unknown_GT_Answer' or \
                            your_response_raw == 'No_Prediction_Available':
                        print(
                            f"Warning: Line {line_number} in {jsonl_file_path} has missing or default data: {stripped_line[:100]}...")

                    # 获取或打开对应的输出文件
                    if output_txt_file_path not in open_files:
                        # 以追加模式打开，这样来自同一个输入文件但分散的同类数据会写入同一个文件
                        # 如果希望每次运行脚本时都覆盖旧的分类文件，可以改为 'w'
                        # 但考虑到可能有多个输入JSONL文件，追加模式通常更安全，除非你知道你在做什么
                        open_files[output_txt_file_path] = open(output_txt_file_path, 'a', encoding='utf-8')
                        print(f"  Writing to new/existing category file: {output_txt_file_path}")

                    outfile = open_files[output_txt_file_path]
                    outfile.write(f"{image_name}\t{question}\t{ground_truth_answer}\t{your_response_cleaned}\n")

                except json.JSONDecodeError:
                    print(f"Skipping malformed JSON line at line {line_number} in {jsonl_file_path}: {line.strip()}")
                except KeyError as e:
                    print(f"Skipping line {line_number} in {jsonl_file_path} due to missing key {e}: {line.strip()}")
                except Exception as e:
                    print(
                        f"An unexpected error occurred at line {line_number} in {jsonl_file_path} for data {line.strip()[:100]}...: {e}")

        print(f"Finished processing {jsonl_file_path}.")

    except FileNotFoundError:
        print(f"Error: Input file '{jsonl_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred during file operations for {jsonl_file_path}: {e}")
    finally:
        # 关闭所有打开的文件
        for f_path, f_obj in open_files.items():
            if f_obj:
                f_obj.close()
                # print(f"Closed file: {f_path}")
        if open_files:
            print(f"All category files for {jsonl_file_path} have been processed and closed.")


if __name__ == "__main__":
    import re  # 需要导入re模块给 sanitize_filename

    # --- 配置 ---
    # 指定存放分类结果的目录，评估脚本要求的是 "Your_Results"
    output_directory = "Your_Results"

    input_jsonl_files = [
        'qwen_mcot_zero0.2.json',
    ]

    # 对于每个输入JSONL文件，进行转换
    for input_jsonl_filename in input_jsonl_files:
        print(f"\n--- Processing input file: {input_jsonl_filename} ---")
        # 注意：这里传递的是输出目录，而不是具体的输出文件名
        convert_jsonl_to_categorized_txt(input_jsonl_filename, output_directory + '/' + input_jsonl_filename.split('.')[0])
        print(f"--- Finished processing for: {input_jsonl_filename} ---")

    print(f"\nAll conversions complete. Categorized .txt files should be in '{output_directory}'.")
