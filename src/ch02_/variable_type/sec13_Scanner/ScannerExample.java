package ch02_.variable_type.sec13_Scanner;

import java.util.Scanner;
// java (compile 방식이면서  interpreter 언어) = 가상머신 
// 객체를 만들어서 넣어주고 작동시킴. 
public class ScannerExample {
	public static void main(String[] args) throws Exception { // 예외발생 시 main(java)던짐. 
		Scanner scanner = new Scanner(System.in); // 표준입력(콘솔창)에서 키보드 입력 
		// java에서 class는 다 new

		System.out.print("x 값 입력: ");
		String strX = scanner.nextLine();
		int x = Integer.parseInt(strX);

		System.out.print("y 값 입력: ");
		String strY = scanner.nextLine();
		int y = Integer.parseInt(strY);

		int result = x + y;
		System.out.println("x + y: " + result);
		System.out.println();

		while(true) {
			System.out.print("입력 문자열: ");
			String data = scanner.nextLine();
			if(data.equals("q")) { // 입력받은 내용이 문자열 "q"이면 무한루프 탈출 
				break;
			}
			System.out.println("출력 문자열: " + data);
			System.out.println();
		}

		System.out.println("종료");
	}
}
/*x 값 입력: 10
y 값 입력: 20
x + y: 30

입력 문자열: 99
출력 문자열: 99

입력 문자열: q
종료
*/