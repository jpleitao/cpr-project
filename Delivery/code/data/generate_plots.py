import csv
import numpy
import matplotlib.pyplot as plt


def generate_plot_imputed_data():
    with open('imputed_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';', quotechar='|')

        count = 0
        time = numpy.array([i for i in range(24)])
        for row in reader:
            if count == 6:
                values = numpy.array(row[1:])
                values_copy = numpy.array(row[1:])
                values = values.astype(numpy.float)
                values_copy = values_copy.astype(numpy.float)

                values_copy[16] = numpy.nan

                plt.plot(time, values, 'b', label='Imputed Data')
                plt.plot(time, values_copy, 'r', label='Original Data')
                plt.legend()
                plt.show()

            count += 1


def generate_elbow_plot_tad():
    with open('metrics_results.csv', 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=';', quotechar='|')

        k_values = list()
        av_wsse = list()

        for row in reader:
            print(row)
            if len(row) > 3 and row[3] == 'TADPole':
                k_values.append(float(row[1]))
                av_wsse.append(float(row[7]))

    fig = plt.figure()
    plt.plot(k_values, av_wsse)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average within-cluster sum of squares')
    fig.savefig('images/elbow_method/raw_dtw_TADPole')
    plt.close(fig)


def main():
    generate_plot_imputed_data()
    generate_elbow_plot_tad()


if __name__ == '__main__':
    main()
