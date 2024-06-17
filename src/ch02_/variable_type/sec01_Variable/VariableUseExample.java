package ch02_.variable_type.sec01_Variable;

public class VariableUseExample {
	public static void main(String[] args) {
		int hour = 3;
		int minute = 5;
		
		// 연산식에 문자열이 포함되어 있으면 숫자를 문자로 변환하여 결합
		System.out.println(hour + "시간 " + minute + "분");

		// int totalMinute = (hour*60) + minute;
		int totalMinute = (hour*60) + minute;
		System.out.println("총 " + totalMinute + "분");
	}
}
