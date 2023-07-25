import java.math.BigInteger;
import java.util.Random;

public class Main
{
    private static void testAdd(int n,int target)
    {
        Random random=new Random(13);
        BigInteger x=new BigInteger(2048,random);
        BigInteger y=new BigInteger(2048,random);
        long start=System.currentTimeMillis();
        for(int i=0;i<n;i++)
        {
            BigInteger c=x.add(y);
        }
        long end=System.currentTimeMillis();
        System.out.println("add cost time:"+((end-start)/1000.0*(target*1.0/n))+"s");
    }

    private static void testSubtract(int n,int target)
    {
        Random random=new Random(13);
        BigInteger x=new BigInteger(2048,random);
        BigInteger y=new BigInteger(2048,random);
        long start=System.currentTimeMillis();
        for(int i=0;i<n;i++)
        {
            BigInteger c=x.subtract(y);
        }
        long end=System.currentTimeMillis();
        System.out.println("subtract cost time:"+((end-start)/1000.0*(target*1.0/n))+"s");
    }

    private static void testMultiply(int n,int target)
    {
        Random random=new Random(13);
        BigInteger x=new BigInteger(2048,random);
        BigInteger y=new BigInteger(2048,random);
        long start=System.currentTimeMillis();
        for(int i=0;i<n;i++)
        {
            BigInteger c=x.multiply(y);
        }
        long end=System.currentTimeMillis();
        System.out.println("multiply cost time:"+((end-start)/1000.0*(target*1.0/n))+"s");
    }

    private static void testDiv(int n,int target)
    {
        Random random=new Random(13);
        BigInteger x=new BigInteger(2048,random);
        BigInteger y=new BigInteger(512,random);
        long start=System.currentTimeMillis();
        for(int i=0;i<n;i++)
        {
            BigInteger c=x.divide(y);
        }
        long end=System.currentTimeMillis();
        System.out.println("div cost time:"+((end-start)/1000.0*(target*1.0/n))+"s");
    }

    private static void testMod(int n,int target)
    {
        Random random=new Random(13);
        BigInteger x=new BigInteger(2048,random);
        BigInteger y=new BigInteger(512,random);
        long start=System.currentTimeMillis();
        for(int i=0;i<n;i++)
        {
            BigInteger c=x.mod(y);
        }
        long end=System.currentTimeMillis();
        System.out.println("mod cost time:"+((end-start)/1000.0*(target*1.0/n))+"s");
    }

    private static void testPowerMod(int n,int target)
    {
        Random random=new Random(13);
        BigInteger x=new BigInteger(1024,random);
        BigInteger y=new BigInteger(1024,random);
        BigInteger z=new BigInteger(1024,random);
        long start=System.currentTimeMillis();
        for(int i=0;i<n;i++)
        {
            BigInteger c=x.modPow(y,z);
        }
        long end=System.currentTimeMillis();
        System.out.println("power_mod cost time:"+((end-start)/1000.0*(target*1.0/n))+"s");
    }

    private static void testRSA(int n,int target)
    {
        Random random=new Random(13);
        BigInteger p=new BigInteger(512,random);
        BigInteger q=new BigInteger(512,random);
        BigInteger m=p.multiply(q);
        BigInteger phi=p.subtract(BigInteger.ONE).multiply(q.subtract(BigInteger.ONE));
        BigInteger e=new BigInteger("65537");
        BigInteger d=e.modInverse(phi);
        BigInteger a=new BigInteger("123456");
        BigInteger c = new BigInteger("0");
        long encrypt_start=System.currentTimeMillis();
        for(int i=0;i<n;i++)
        {
            c=a.modPow(e,m);
        }
        long encrypt_end=System.currentTimeMillis();
        System.out.println("rsa encrypt cost time:"+((encrypt_end-encrypt_start)/1000.0*(target*1.0/n))+"s");

        long decrypt_start=System.currentTimeMillis();
        BigInteger b;
        for(int i=0;i<n;i++)
        {
            b=c.modPow(d,m);
        }
        long decrypt_end=System.currentTimeMillis();
        System.out.println("rsa decrypt cost time:"+((decrypt_end-decrypt_start)/1000.0*(target*1.0/n))+"s");
    }

    public static void main(String[] args)
    {
        testAdd(42607141,100000000);
        testSubtract(42607141,100000000);
        testMultiply(42607141,100000000);
        testDiv(42607141,100000000);
        testMod(42607141,100000000);
        testPowerMod(6833,1000000);
        testRSA(6833,1000000);
    }
}
