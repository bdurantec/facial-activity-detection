from report.report_dataclass import ReportDataclass


class ReportGenerator:
    def __init__(self, report_path):
        self.__report_path = report_path
        self.__content = []
        self.__initialize()

    def __initialize(self):
        self.__content.append('# Video Analysis Report\n')

    def add_report(self, report: ReportDataclass):
        content = '''
## {} 
Total Frames Analyzed: {}
Number of Anomalies Detected: {}

### Summary
{}

        '''.format(
            report.title,
            report.total_frames,
            report.anomalies_detected,
            report.summary
        )

        self.__content.append(content)

    def finish(self):
        with open(self.__report_path, 'w') as report_file:
            for text in self.__content:
                report_file.write(text)

            report_file.close()
        print(f'Report generated and saved to: {self.__report_path}')
