package ch05_.references.sec11.MainStringArgument;
//IDE : run -> configuration -> argument 에 90 100 입력하면 190 출력
// E:\JWorkspace\Workspace\thisisjava\bin\ch05_\references\sec11\MainStringArgument

public class MainStringArrayArgument {
	public static void main(String[] args) {
		if(args.length != 2) {
			System.out.println("프로그램 입력값이 부족");
			System.out.println("Usage : 인자1 인자2");
	
			System.exit(0);
		}

		String strNum1 = args[0];S
		String strNum2 = args[1];
			
		int num1 = Integer.parseInt(strNum1);
		int num2 = Integer.parseInt(strNum2);

		int result = num1 + num2;
		System.out.println(num1 + " + " + num2 + " = " + result);
	}
}