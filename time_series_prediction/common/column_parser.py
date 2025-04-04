import re

class ColumnParser:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def _get_all_columns(self):
        return list(self.dataframe.columns)

    def parse_columns(self, columns):
        all_columns = self._get_all_columns()

        if isinstance(columns, list):
            columns = ' '.join(str(col) for col in columns)

        if isinstance(columns, str):
            if columns.startswith('@all'):
                exclude_spec = re.findall(r'\[(.*?)\]', columns)
                if exclude_spec:
                    exclude_columns = exclude_spec[0].split(',')
                    exclude_columns = [col.strip() for col in exclude_columns]

                    exclude_indices = []
                    exclude_names = []

                    for item in exclude_columns:
                        if item.isdigit():
                            exclude_indices.append(int(item))
                        else:
                            exclude_names.append(item)

                    exclude_columns = []
                    if exclude_indices:
                        exclude_columns.extend([all_columns[i] for i in exclude_indices if i < len(all_columns)])
                    if exclude_names:
                        exclude_columns.extend([name for name in exclude_names if name in all_columns])

                    return [col for col in all_columns if col not in exclude_columns]
                else:
                    return all_columns
            elif columns.startswith('[') and columns.endswith(']'):
                columns = columns[1:-1].split(',')
                result_columns = []
                for col in columns:
                    col = col.strip()
                    if col.isdigit():
                        index = int(col)
                        if index < len(all_columns):
                            result_columns.append(all_columns[index])
                    elif col in all_columns:
                        result_columns.append(col)
                return result_columns
            else:
                column_mapping = {}
                for col in columns.split():
                    if ':' in col:
                        name, num = col.split(':')
                        column_mapping[name.strip()] = int(num.strip())
                    else:
                        column_mapping[col.strip()] = None

                if None in column_mapping.values():
                    included_columns = [name for name, num in column_mapping.items() if num is None]
                    return included_columns
                else:
                    return [name for name, num in column_mapping.items()]
        else:
            raise TypeError("Columns must be a string or list.")

