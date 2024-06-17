package ch11_exception.sec02.exam02;
// 일반 예외(exception) : compiler 가 예외 처리코드 여부를 검사하는 예외
// ClassNotFoundException, interruptedException
public class ExceptionHandlingExample {
	
	public static void main(String[] args) {
		try {
			Class.forName("java.lang.String");
			System.out.println("java.lang.String 클래스가 존재합니다.");
		} catch(ClassNotFoundException e) {
			e.printStackTrace();
		}

		System.out.println();

		try {
			Class.forName("java.lang.String2");
			// 예외가 발생하면 하위 명령문은 실행되지 않음 
			System.out.println("java.lang.String2 클래스가 존재합니다.");
		} catch(ClassNotFoundException e) {
			//e.printStackTrace();
		}
		
		System.out.println("정상 종료");
	}
}