

package ch03_.operator.sec08_bitlogical;
// 1-> 2- > 4 -> 8 비트가 증가할 떄 마다 0000 0000 -> 0000 0010 자릿수 증가
public class BitLogicExample {
	public static void main(String[] args) {
		// &  : 둘다 1인 경우 1 이됨./ | 하나라도 1이면 1  / ^ 서로 다르면 1 / ~ 각 비트 반전 1->0 / 0->1
		System.out.println("45 & 25 = " + (45 & 25)); // 0010 1101 & 0001 1001 -> 0000 1001 -> 9
		System.out.println("45 | 25 = " + (45 | 25)); // 0010 1101 & 0001 1001 -> 32+16+8+4+0+1 -> 61
		System.out.println("45 ^ 25 = " + (45 ^ 25)); // 0010 1101 & 0001 1001 -> 32 + 16 + 0 + 4 -> 52
		System.out.println("~45 = " + (~45)); // 1101 0010; -> -46 [ 음수의 경우 절댓값을 취해주고 1을 더해주어야함]
		System.out.println("~1 = " + (~1)); // 0000 0001 -> 1111 1110 -> -2
		
		System.out.println("-------------------------------");

		byte receiveData = -120; // 1000 1000

		//방법1: 비트 논리곱 연산으로 Unsigned 정수 얻기
		int unsignedInt1 = receiveData & 255; // 1111 1111 // (int : 4byte)
		System.out.println(unsignedInt1); // 136

		//방법2: 자바 API를 이용해서 Unsigned 정수 얻기
		int unsignedInt2 = Byte.toUnsignedInt(receiveData);
		System.out.println(unsignedInt2); // 136

		int test = 136; // 1000 1000 자바에서는 최상위 비트가 1 | 000 1000 1이면 음수로 인식
		byte btest = (byte) test; 
		System.out.println(btest); // -120 
	}
}