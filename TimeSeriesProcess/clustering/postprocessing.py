# coding: utf-8

"""
Stores useful post-processing functions, to gather information about the obtained clusters
"""

import datetime

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


# TODO: Método que recebe os assignments dos dias a cada cluster e retorna um dicionario ou assim que diga quantos dias
# de semana tem cada cluster, qual a percentagem de dias de cada mês, etc

# FIXME: Mudar isto para o module utils???


def is_weekday(day_date):
    """
    Receives a given date as a datetime.datetime type, and returns a boolean variable, signalling whether or not the
    date in question is a weekday.
    :param day_date: The date to be processed, as a datetime.datetime type
    :return: A boolean variable, signalling whether or not the date in question is a weekday.
    """
    if not isinstance(day_date, datetime.datetime):
        return TypeError('Argument <day_date> should be of type <datetime.datetime>!')

    # datetime.date.weekday() - "Return day of the week, where Monday == 0 ... Sunday == 6."
    # So it is weekend if it returns 5 or 6, a lower value indicates weekday
    return day_date.weekday() < 5
