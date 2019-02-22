import random


class RandomColumnSelector:
    def __init__(self, base_col, try_col, base_col_prob_list, try_col_prob_list):
        self.base_col = base_col
        self.try_col = try_col
        self.base_col_prob_list = base_col_prob_list
        self.try_col_prob_list = try_col_prob_list

    def select_col(self, seed):
        random.seed(seed)
        use_base_col, use_base_col_prob = self._get_use_cols(self.base_col_prob_list, self.base_col)
        use_try_col, use_try_col_prob = self._get_use_cols(self.try_col_prob_list, self.try_col)

        ret_col = use_base_col + use_try_col
        return ret_col, use_base_col_prob, use_try_col_prob

    @staticmethod
    def _get_use_cols(prob_list, col_list):
        use_prob = random.choice(prob_list)
        use_col_samples = round(len(col_list) * use_prob)
        use_col = random.sample(col_list, use_col_samples)
        return use_col, use_prob


# x = select_col(["a", "b"], ["c"], 0.6, 0.3)
# print(x)
#
#
