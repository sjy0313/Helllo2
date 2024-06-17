package ch04_.statement.sec06.dowhileloop;

import java.util.Scanner;
// do ~ while 문 : 블록을 한 번은 실행하고 조건식을 확인, 조건식이 참이면 다시 반복 
// 데이터 처리에 유용하게 사용가능
public class DoWhileExample {
	public static void main(String[] args) {
		System.out.println("메시지를 입력하세요.");
		System.out.println("프로그램을 종료하려면 q를 입력하세요.");

		Scanner scanner = new Scanner(System.in);
		String inputString;

		do { // 반드시 한번은 사용
			
			System.out.print(">");
			inputString = scanner.nextLine();
			System.out.println(inputString);
		} while( ! inputString.equals("q") ); //'q'가 아니라면 지속/ 'q'라면 종료

		System.out.println();
		System.out.println("프로그램 종료");
	}
}
