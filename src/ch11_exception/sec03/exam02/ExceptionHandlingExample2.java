package ch11_exception.sec03.exam02;

public class ExceptionHandlingExample2 {
	public static void main(String[] args) {
		String[] array = {"100", "1oo"};

		for(int i=0; i<=array.length; i++) {
			try {
				int value = Integer.parseInt(array[i]);
				System.out.println("array[" + i + "]: " + value);
			} catch(Exception e) {
					System.out.println("실행에 문제가 있습니다.");
				
			} 
			/* catch에서 exception이 앞에 나오면 아래와 같은 오류가 나옴. 
			
			catch(ArrayIndexOutOfBoundsException e) {
				System.out.println("배열 인덱스가 초과됨: " + e.getMessage());
			} 
			*/
		}
		
	}
}